/*
 * lfm2_gui.c — Native C11 X11 + Xft Desktop GUI for LFM2 / LFM3 Models
 *
 * Uses include/lfm_ui.h (single-header immediate-mode UI) and
 * tools/lfm2_gui_worker.h (local liblfm2.so engine + optional SSE server client).
 *
 * Zero external GUI toolkit or webview dependencies: pure Xlib + Xft + pthreads.
 */

#define LFM_UI_IMPLEMENTATION
#include "lfm_ui.h"
#include "lfm2_gui_worker.h"

#include <X11/Xatom.h>
#include <X11/Xft/Xft.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/keysym.h>
#include <fcntl.h>
#include <locale.h>
#include <poll.h>
#include <signal.h>

/* ============================================================================
 * 1. X11 + Xft Rendering & Clipboard Backend
 * ============================================================================ */

typedef struct {
    Display  *dpy;
    int       screen;
    Window    win;
    Pixmap    backbuf;
    int32_t   buf_w;
    int32_t   buf_h;
    GC        gc;
    Visual   *visual;
    Colormap  colormap;
    XftDraw  *xft_draw;
    XftFont  *fonts[LFM_UI_FONT_COUNT];
    Cursor    cursors[3];
    LfmUiCursorKind active_cursor;

    XIM       xim;
    XIC       xic;

    Atom      wm_delete;
    Atom      atom_clipboard;
    Atom      atom_utf8;
    Atom      atom_targets;
    Atom      atom_prop;

    char     *clip_owned;
    size_t    clip_owned_len;

    int       wake_pipe[2];
    bool      should_quit;
} X11Backend;

static unsigned long x11_pixel(X11Backend *x, LfmUiColor c) {
    (void)x;
    return ((unsigned long)c.r << 16) | ((unsigned long)c.g << 8) | (unsigned long)c.b;
}

static uint64_t x11_now_ms(void *ud) {
    (void)ud;
    return lfm_gui_now_ms();
}

static void x11_ensure_backbuf(X11Backend *x, int32_t w, int32_t h) {
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    if (x->backbuf && x->buf_w == w && x->buf_h == h) return;

    if (x->xft_draw) {
        XftDrawDestroy(x->xft_draw);
        x->xft_draw = NULL;
    }
    if (x->backbuf) {
        XFreePixmap(x->dpy, x->backbuf);
        x->backbuf = None;
    }
    x->backbuf = XCreatePixmap(x->dpy, x->win, (unsigned int)w, (unsigned int)h,
                               (unsigned int)DefaultDepth(x->dpy, x->screen));
    x->xft_draw = XftDrawCreate(x->dpy, x->backbuf, x->visual, x->colormap);
    x->buf_w = w;
    x->buf_h = h;
}

static void x11_set_clip(void *ud, LfmUiRect r) {
    X11Backend *x = (X11Backend *)ud;
    XRectangle xr;
    xr.x = (short)(r.x < -32768 ? -32768 : r.x);
    xr.y = (short)(r.y < -32768 ? -32768 : r.y);
    xr.width = (unsigned short)(r.w > 0 ? r.w : 0);
    xr.height = (unsigned short)(r.h > 0 ? r.h : 0);
    XSetClipRectangles(x->dpy, x->gc, 0, 0, &xr, 1, Unsorted);
    if (x->xft_draw) {
        XftDrawSetClipRectangles(x->xft_draw, 0, 0, &xr, 1);
    }
}

static void x11_draw_rect(void *ud, LfmUiRect r, LfmUiColor c) {
    X11Backend *x = (X11Backend *)ud;
    if (c.a == 0 || r.w <= 0 || r.h <= 0) return;
    XSetForeground(x->dpy, x->gc, x11_pixel(x, c));
    XFillRectangle(x->dpy, x->backbuf, x->gc, r.x, r.y, (unsigned int)r.w, (unsigned int)r.h);
}

static void x11_fill_rounded(Display *d, Drawable dr, GC gc, LfmUiRect r, int32_t rad) {
    if (rad <= 0) {
        XFillRectangle(d, dr, gc, r.x, r.y, (unsigned int)r.w, (unsigned int)r.h);
        return;
    }
    int32_t max_r = (r.w < r.h ? r.w : r.h) / 2;
    if (rad > max_r) rad = max_r;
    int32_t d2 = rad * 2;

    XFillRectangle(d, dr, gc, r.x + rad, r.y, (unsigned int)(r.w - d2), (unsigned int)r.h);
    XFillRectangle(d, dr, gc, r.x, r.y + rad, (unsigned int)rad, (unsigned int)(r.h - d2));
    XFillRectangle(d, dr, gc, r.x + r.w - rad, r.y + rad, (unsigned int)rad, (unsigned int)(r.h - d2));

    XFillArc(d, dr, gc, r.x, r.y, (unsigned int)d2, (unsigned int)d2, 90 * 64, 90 * 64);
    XFillArc(d, dr, gc, r.x + r.w - d2, r.y, (unsigned int)d2, (unsigned int)d2, 0, 90 * 64);
    XFillArc(d, dr, gc, r.x, r.y + r.h - d2, (unsigned int)d2, (unsigned int)d2, 180 * 64, 90 * 64);
    XFillArc(d, dr, gc, r.x + r.w - d2, r.y + r.h - d2, (unsigned int)d2, (unsigned int)d2, 270 * 64, 90 * 64);
}

static void x11_stroke_rounded(Display *d, Drawable dr, GC gc, LfmUiRect r, int32_t rad) {
    if (r.w < 2 || r.h < 2) return;
    int32_t w = r.w - 1, h = r.h - 1;
    if (rad <= 0) {
        XDrawRectangle(d, dr, gc, r.x, r.y, (unsigned int)w, (unsigned int)h);
        return;
    }
    int32_t max_r = (w < h ? w : h) / 2;
    if (rad > max_r) rad = max_r;
    int32_t d2 = rad * 2;

    XDrawLine(d, dr, gc, r.x + rad, r.y, r.x + w - rad, r.y);
    XDrawLine(d, dr, gc, r.x + rad, r.y + h, r.x + w - rad, r.y + h);
    XDrawLine(d, dr, gc, r.x, r.y + rad, r.x, r.y + h - rad);
    XDrawLine(d, dr, gc, r.x + w, r.y + rad, r.x + w, r.y + h - rad);

    XDrawArc(d, dr, gc, r.x, r.y, (unsigned int)d2, (unsigned int)d2, 90 * 64, 90 * 64);
    XDrawArc(d, dr, gc, r.x + w - d2, r.y, (unsigned int)d2, (unsigned int)d2, 0, 90 * 64);
    XDrawArc(d, dr, gc, r.x, r.y + h - d2, (unsigned int)d2, (unsigned int)d2, 180 * 64, 90 * 64);
    XDrawArc(d, dr, gc, r.x + w - d2, r.y + h - d2, (unsigned int)d2, (unsigned int)d2, 270 * 64, 90 * 64);
}

static void x11_draw_rounded_rect(void *ud, LfmUiRect r, int32_t radius,
                                  LfmUiColor fill, LfmUiColor border, int32_t border_w) {
    X11Backend *x = (X11Backend *)ud;
    if (r.w <= 0 || r.h <= 0) return;
    if (fill.a > 0) {
        XSetForeground(x->dpy, x->gc, x11_pixel(x, fill));
        x11_fill_rounded(x->dpy, x->backbuf, x->gc, r, radius);
    }
    if (border_w > 0 && border.a > 0) {
        XSetForeground(x->dpy, x->gc, x11_pixel(x, border));
        x11_stroke_rounded(x->dpy, x->backbuf, x->gc, r, radius);
    }
}

static XftFont *x11_font(X11Backend *x, LfmUiFontRole role) {
    if ((int)role < 0 || (int)role >= LFM_UI_FONT_COUNT) role = LFM_UI_FONT_UI;
    return x->fonts[role] ? x->fonts[role] : x->fonts[LFM_UI_FONT_UI];
}

static int32_t x11_measure_text(void *ud, LfmUiFontRole role, const char *text, size_t len) {
    X11Backend *x = (X11Backend *)ud;
    XftFont *f = x11_font(x, role);
    if (!f || !text || len == 0) return 0;
    XGlyphInfo ext;
    XftTextExtentsUtf8(x->dpy, f, (const FcChar8 *)text, (int)len, &ext);
    return (int32_t)ext.xOff;
}

static int32_t x11_line_height(void *ud, LfmUiFontRole role) {
    X11Backend *x = (X11Backend *)ud;
    XftFont *f = x11_font(x, role);
    if (!f) return 18;
    int32_t lh = (int32_t)(f->ascent + f->descent);
    return lh > 0 ? lh : 18;
}

static void x11_draw_text(void *ud, LfmUiFontRole role, int32_t px, int32_t py,
                          LfmUiColor c, const char *text, size_t len) {
    X11Backend *x = (X11Backend *)ud;
    XftFont *f = x11_font(x, role);
    if (!f || !x->xft_draw || !text || len == 0) return;

    XRenderColor rc;
    rc.red = (unsigned short)(c.r * 257);
    rc.green = (unsigned short)(c.g * 257);
    rc.blue = (unsigned short)(c.b * 257);
    rc.alpha = (unsigned short)(c.a * 257);

    XftColor xc;
    if (XftColorAllocValue(x->dpy, x->visual, x->colormap, &rc, &xc)) {
        XftDrawStringUtf8(x->xft_draw, &xc, f, px, py + f->ascent, (const FcChar8 *)text, (int)len);
        XftColorFree(x->dpy, x->visual, x->colormap, &xc);
    }
}

static void x11_set_cursor(void *ud, LfmUiCursorKind kind) {
    X11Backend *x = (X11Backend *)ud;
    if (x->active_cursor == kind) return;
    x->active_cursor = kind;
    XDefineCursor(x->dpy, x->win, x->cursors[(int)kind]);
}

static void x11_clipboard_set(void *ud, const char *text, size_t len) {
    X11Backend *x = (X11Backend *)ud;
    if (!text) return;
    free(x->clip_owned);
    x->clip_owned = (char *)malloc(len + 1);
    if (!x->clip_owned) return;
    memcpy(x->clip_owned, text, len);
    x->clip_owned[len] = '\0';
    x->clip_owned_len = len;
    XSetSelectionOwner(x->dpy, x->atom_clipboard, x->win, CurrentTime);
    XSetSelectionOwner(x->dpy, XA_PRIMARY, x->win, CurrentTime);
    XFlush(x->dpy);
}

static size_t x11_clipboard_get(void *ud, char *out, size_t cap) {
    X11Backend *x = (X11Backend *)ud;
    if (!out || cap == 0) return 0;

    if (XGetSelectionOwner(x->dpy, x->atom_clipboard) == x->win && x->clip_owned) {
        size_t n = x->clip_owned_len < cap ? x->clip_owned_len : cap - 1;
        memcpy(out, x->clip_owned, n);
        out[n] = '\0';
        return n;
    }

    XConvertSelection(x->dpy, x->atom_clipboard, x->atom_utf8, x->atom_prop, x->win, CurrentTime);
    XFlush(x->dpy);

    uint64_t deadline = lfm_gui_now_ms() + 250;
    while (lfm_gui_now_ms() < deadline) {
        XEvent ev;
        if (XCheckTypedWindowEvent(x->dpy, x->win, SelectionNotify, &ev)) {
            if (ev.xselection.property == None) return 0;
            Atom actual_type = None;
            int actual_fmt = 0;
            unsigned long nitems = 0, bytes_after = 0;
            unsigned char *prop = NULL;
            if (XGetWindowProperty(x->dpy, x->win, x->atom_prop, 0, 65536, True,
                                   AnyPropertyType, &actual_type, &actual_fmt,
                                   &nitems, &bytes_after, &prop) == Success && prop) {
                size_t copy = nitems < cap ? (size_t)nitems : (cap - 1);
                memcpy(out, prop, copy);
                out[copy] = '\0';
                XFree(prop);
                return copy;
            }
            return 0;
        }
        usleep(2000);
    }
    return 0;
}

static void x11_handle_selection_request(X11Backend *x, const XSelectionRequestEvent *req) {
    XSelectionEvent resp;
    memset(&resp, 0, sizeof(resp));
    resp.type = SelectionNotify;
    resp.display = req->display;
    resp.requestor = req->requestor;
    resp.selection = req->selection;
    resp.target = req->target;
    resp.time = req->time;
    resp.property = None;

    if (req->target == x->atom_targets) {
        Atom list[2] = {x->atom_utf8, XA_STRING};
        XChangeProperty(x->dpy, req->requestor, req->property, XA_ATOM, 32,
                        PropModeReplace, (unsigned char *)list, 2);
        resp.property = req->property;
    } else if ((req->target == x->atom_utf8 || req->target == XA_STRING) && x->clip_owned) {
        XChangeProperty(x->dpy, req->requestor, req->property, req->target, 8,
                        PropModeReplace, (unsigned char *)x->clip_owned, (int)x->clip_owned_len);
        resp.property = req->property;
    }

    XSendEvent(x->dpy, req->requestor, False, 0, (XEvent *)&resp);
}

static XftFont *x11_open_font(Display *dpy, int screen, const char *pattern) {
    XftFont *f = XftFontOpenName(dpy, screen, pattern);
    if (!f) f = XftFontOpenName(dpy, screen, "sans-10");
    if (!f) f = XftFontOpenName(dpy, screen, "fixed");
    return f;
}

static bool x11_init(X11Backend *x, int32_t w, int32_t h) {
    memset(x, 0, sizeof(*x));
    x->wake_pipe[0] = -1;
    x->wake_pipe[1] = -1;
    if (pipe(x->wake_pipe) != 0) return false;
    for (int i = 0; i < 2; ++i) {
        int fl = fcntl(x->wake_pipe[i], F_GETFL, 0);
        fcntl(x->wake_pipe[i], F_SETFL, fl | O_NONBLOCK);
    }

    x->dpy = XOpenDisplay(NULL);
    if (!x->dpy) return false;

    x->screen = DefaultScreen(x->dpy);
    x->visual = DefaultVisual(x->dpy, x->screen);
    x->colormap = DefaultColormap(x->dpy, x->screen);

    XSetWindowAttributes swa;
    swa.background_pixel = 0x0F1117;
    swa.event_mask = ExposureMask | StructureNotifyMask | KeyPressMask |
                     ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
                     FocusChangeMask;
    x->win = XCreateWindow(x->dpy, RootWindow(x->dpy, x->screen), 80, 60,
                           (unsigned int)w, (unsigned int)h, 0,
                           CopyFromParent, InputOutput, CopyFromParent,
                           CWBackPixel | CWEventMask, &swa);

    XStoreName(x->dpy, x->win, "LFM Studio — Native C11 Desktop");
    XSizeHints *sh = XAllocSizeHints();
    if (sh) {
        sh->flags = PMinSize;
        sh->min_width = 720;
        sh->min_height = 520;
        XSetWMNormalHints(x->dpy, x->win, sh);
        XFree(sh);
    }

    x->wm_delete = XInternAtom(x->dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(x->dpy, x->win, &x->wm_delete, 1);
    x->atom_clipboard = XInternAtom(x->dpy, "CLIPBOARD", False);
    x->atom_utf8 = XInternAtom(x->dpy, "UTF8_STRING", False);
    x->atom_targets = XInternAtom(x->dpy, "TARGETS", False);
    x->atom_prop = XInternAtom(x->dpy, "LFM_CLIP_DATA", False);

    x->gc = XCreateGC(x->dpy, x->win, 0, NULL);
    x->fonts[LFM_UI_FONT_UI] = x11_open_font(x->dpy, x->screen, "sans-10");
    x->fonts[LFM_UI_FONT_BOLD] = x11_open_font(x->dpy, x->screen, "sans-10:bold");
    x->fonts[LFM_UI_FONT_MONO] = x11_open_font(x->dpy, x->screen, "monospace-10");
    x->fonts[LFM_UI_FONT_SMALL] = x11_open_font(x->dpy, x->screen, "sans-8.5");
    x->fonts[LFM_UI_FONT_TITLE] = x11_open_font(x->dpy, x->screen, "sans-12:bold");

    x->cursors[LFM_UI_CURSOR_ARROW] = XCreateFontCursor(x->dpy, XC_left_ptr);
    x->cursors[LFM_UI_CURSOR_IBEAM] = XCreateFontCursor(x->dpy, XC_xterm);
    x->cursors[LFM_UI_CURSOR_HAND] = XCreateFontCursor(x->dpy, XC_hand2);

    x->xim = XOpenIM(x->dpy, NULL, NULL, NULL);
    if (x->xim) {
        x->xic = XCreateIC(x->xim, XNInputStyle, XIMPreeditNothing | XIMStatusNothing,
                           XNClientWindow, x->win, XNFocusWindow, x->win, NULL);
    }

    x11_ensure_backbuf(x, w, h);
    XMapWindow(x->dpy, x->win);
    XFlush(x->dpy);
    return true;
}

static void x11_destroy(X11Backend *x) {
    free(x->clip_owned);
    if (x->xic) XDestroyIC(x->xic);
    if (x->xim) XCloseIM(x->xim);
    for (int i = 0; i < 3; ++i) {
        if (x->cursors[i]) XFreeCursor(x->dpy, x->cursors[i]);
    }
    for (int i = 0; i < LFM_UI_FONT_COUNT; ++i) {
        if (x->fonts[i]) XftFontClose(x->dpy, x->fonts[i]);
    }
    if (x->xft_draw) XftDrawDestroy(x->xft_draw);
    if (x->backbuf) XFreePixmap(x->dpy, x->backbuf);
    if (x->gc) XFreeGC(x->dpy, x->gc);
    if (x->win) XDestroyWindow(x->dpy, x->win);
    if (x->dpy) XCloseDisplay(x->dpy);
    if (x->wake_pipe[0] >= 0) close(x->wake_pipe[0]);
    if (x->wake_pipe[1] >= 0) close(x->wake_pipe[1]);
}

static void x11_wake_callback(void *ud) {
    X11Backend *x = (X11Backend *)ud;
    if (x && x->wake_pipe[1] >= 0) {
        char b = 1;
        ssize_t unused = write(x->wake_pipe[1], &b, 1);
        (void)unused;
    }
}

static LfmUiKey translate_keysym(KeySym ks) {
    switch (ks) {
        case XK_BackSpace: return LFM_UI_KEY_BACKSPACE;
        case XK_Delete:    return LFM_UI_KEY_DELETE;
        case XK_Left:      return LFM_UI_KEY_LEFT;
        case XK_Right:     return LFM_UI_KEY_RIGHT;
        case XK_Up:        return LFM_UI_KEY_UP;
        case XK_Down:      return LFM_UI_KEY_DOWN;
        case XK_Home:      return LFM_UI_KEY_HOME;
        case XK_End:       return LFM_UI_KEY_END;
        case XK_Return:
        case XK_KP_Enter:  return LFM_UI_KEY_ENTER;
        case XK_Tab:       return LFM_UI_KEY_TAB;
        case XK_Escape:    return LFM_UI_KEY_ESCAPE;
        default:           return LFM_UI_KEY_NONE;
    }
}

/* ============================================================================
 * 2. Application State & Layout Rendering
 * ============================================================================ */

typedef struct {
    LfmUiContext   ui;
    X11Backend     x11;
    LfmGuiEngine   eng;
    LfmGuiConfig   cfg;

    LfmUiScroll    chat_scroll;
    LfmUiScroll    modal_scroll;
    LfmUiTextEdit  composer;

    bool           dark_theme;
    bool           show_settings;
    bool           show_model_modal;
    char           copied_toast[64];
    uint64_t       copied_until_ms;

    char           modal_dir[512];
    LfmGuiFileEntry modal_files[128];
    size_t         modal_count;
} LfmApp;

static void app_refresh_modal_dir(LfmApp *app, const char *dir) {
    snprintf(app->modal_dir, sizeof(app->modal_dir), "%s", dir);
    app->modal_count = lfm_gui_scan_dir(app->modal_dir, app->modal_files, 128);
    app->modal_scroll.offset_y = 0;
}

static void app_submit_text(LfmApp *app, const char *prompt) {
    if (!prompt || !prompt[0]) return;
    if (lfm_gui_engine_send_prompt(&app->eng, prompt)) {
        lfm_ui_scroll_to_bottom(&app->chat_scroll);
        lfm_ui_textedit_clear(&app->composer);
    }
}

static void app_render_header(LfmApp *app, LfmUiRect r) {
    LfmUiContext *ui = &app->ui;
    lfm_ui_draw_rect(ui, r, ui->theme.bg_header);
    lfm_ui_draw_rect(ui, lfm_rect(r.x, r.y + r.h - 1, r.w, 1), ui->theme.border);

    LfmUiRect row = lfm_rect_inset(r, 12, 9);

    /* Title badge */
    lfm_ui_draw_text_cstr(ui, LFM_UI_FONT_TITLE, row.x, row.y + 4, ui->theme.text, "LFM Studio");
    int32_t title_w = lfm_ui_text_w(ui, LFM_UI_FONT_TITLE, "LFM Studio", 10) + 16;
    lfm_rect_cut_left(&row, title_w);

    /* Right controls */
    LfmUiRect btn_set = lfm_rect_cut_right(&row, 84);
    lfm_rect_cut_right(&row, 8);
    LfmUiRect btn_theme = lfm_rect_cut_right(&row, 68);
    lfm_rect_cut_right(&row, 8);
    LfmUiRect btn_new = lfm_rect_cut_right(&row, 88);
    lfm_rect_cut_right(&row, 8);
    LfmUiRect btn_mode = lfm_rect_cut_right(&row, 118);
    lfm_rect_cut_right(&row, 10);

    if (lfm_ui_button(ui, lfm_ui_hash_str("hdr_settings"), btn_set,
                      app->show_settings ? "Settings*" : "Settings", app->show_settings, true)) {
        app->show_settings = !app->show_settings;
    }
    if (lfm_ui_button(ui, lfm_ui_hash_str("hdr_theme"), btn_theme,
                      app->dark_theme ? "Light" : "Dark", false, true)) {
        app->dark_theme = !app->dark_theme;
        ui->theme = app->dark_theme ? lfm_ui_theme_dark() : lfm_ui_theme_light();
    }
    if (lfm_ui_button(ui, lfm_ui_hash_str("hdr_new"), btn_new, "New Chat", false, true)) {
        lfm_gui_engine_reset_chat(&app->eng);
        app->chat_scroll.offset_y = 0;
    }

    bool is_server = (app->cfg.backend_mode == LFM_BACKEND_SERVER);
    if (lfm_ui_button(ui, lfm_ui_hash_str("hdr_mode"), btn_mode,
                      is_server ? "Mode: Server" : "Mode: Local", is_server, true)) {
        pthread_mutex_lock(&app->eng.mu);
        app->cfg.backend_mode = is_server ? LFM_BACKEND_LOCAL : LFM_BACKEND_SERVER;
        app->eng.cfg.backend_mode = app->cfg.backend_mode;
        if (app->cfg.backend_mode == LFM_BACKEND_SERVER) {
            snprintf(app->eng.status_text, sizeof(app->eng.status_text),
                     "Server %.128s:%d", app->cfg.server_host, app->cfg.server_port);
        } else {
            snprintf(app->eng.status_text, sizeof(app->eng.status_text), "Local %.200s",
                     app->cfg.model_path[0] ? lfm_gui_basename(app->cfg.model_path) : "(no model)");
        }
        pthread_mutex_unlock(&app->eng.mu);
    }

    /* Model picker button + live status label */
    int32_t picker_w = row.w > 340 ? 270 : (row.w > 180 ? row.w - 80 : row.w);
    if (picker_w > 90) {
        LfmUiRect picker_r = lfm_rect_cut_left(&row, picker_w);
        const char *mname = app->cfg.model_path[0] ? lfm_gui_basename(app->cfg.model_path) : "Select .gguf Model...";
        char lbl[160];
        snprintf(lbl, sizeof(lbl), "%.42s  ▾", mname);
        if (lfm_ui_button(ui, lfm_ui_hash_str("hdr_model"), picker_r, lbl, false, true)) {
            app->show_model_modal = !app->show_model_modal;
            if (app->show_model_modal) {
                app_refresh_modal_dir(app, app->modal_dir[0] ? app->modal_dir : "models");
            }
        }
    }

    if (row.w > 80) {
        lfm_rect_cut_left(&row, 10);
        pthread_mutex_lock(&app->eng.mu);
        LfmGuiStatus st = app->eng.status;
        char st_txt[256];
        snprintf(st_txt, sizeof(st_txt), "%s", app->eng.status_text);
        pthread_mutex_unlock(&app->eng.mu);
        LfmUiColor sc = (st == LFM_GUI_ERROR) ? ui->theme.danger :
                        (st == LFM_GUI_GENERATING ? ui->theme.accent :
                         (st == LFM_GUI_LOADING_MODEL ? ui->theme.warn : ui->theme.success));
        lfm_ui_draw_card(ui, lfm_rect(row.x, row.y + 14, 8, 8), 4, sc, lfm_rgba(0, 0, 0, 0));
        lfm_ui_draw_text_clipped(ui, LFM_UI_FONT_SMALL, lfm_rect(row.x + 14, row.y, row.w - 14, row.h),
                                 ui->theme.text_muted, st_txt);
    }
}

static void app_render_settings_drawer(LfmApp *app, LfmUiRect r) {
    LfmUiContext *ui = &app->ui;
    lfm_ui_draw_rect(ui, r, ui->theme.bg_panel);
    lfm_ui_draw_rect(ui, lfm_rect(r.x, r.y, 1, r.h), ui->theme.border);

    LfmUiRect inner = lfm_rect_inset(r, 14, 14);
    lfm_ui_draw_text_cstr(ui, LFM_UI_FONT_BOLD, inner.x, inner.y, ui->theme.text, "Generation & Runtime");
    lfm_rect_cut_top(&inner, 28);

    bool dirty = false;

    dirty |= lfm_ui_slider_f(ui, lfm_ui_hash_str("set_temp"), lfm_rect_cut_top(&inner, 44),
                             "Temperature", &app->cfg.temperature, 0.0f, 1.5f, "%.2f");
    lfm_rect_cut_top(&inner, 8);
    dirty |= lfm_ui_slider_f(ui, lfm_ui_hash_str("set_minp"), lfm_rect_cut_top(&inner, 44),
                             "Min-P", &app->cfg.min_p, 0.0f, 0.5f, "%.2f");
    lfm_rect_cut_top(&inner, 8);
    dirty |= lfm_ui_slider_i(ui, lfm_ui_hash_str("set_topk"), lfm_rect_cut_top(&inner, 44),
                             "Top-K", &app->cfg.top_k, 0, 100, "");
    lfm_rect_cut_top(&inner, 10);

    char max_lbl[64];
    snprintf(max_lbl, sizeof(max_lbl), "Max Tokens (%s)", app->cfg.max_tokens == 0 ? "∞" : "cap");
    dirty |= lfm_ui_slider_i(ui, lfm_ui_hash_str("set_maxtok"), lfm_rect_cut_top(&inner, 44),
                             max_lbl, &app->cfg.max_tokens, 0, 8000, "");
    lfm_rect_cut_top(&inner, 4);

    /* Quick max_tokens chips */
    {
        static const int32_t presets[4] = {512, 2000, 4096, 0};
        static const char *labels[4] = {"512", "2000", "4096", "∞"};
        LfmUiRect row = lfm_rect_cut_top(&inner, 26);
        int32_t cw = (row.w - 18) / 4;
        for (int i = 0; i < 4; ++i) {
            LfmUiRect cr = lfm_rect(row.x + i * (cw + 6), row.y, cw, 24);
            if (lfm_ui_chip(ui, lfm_ui_hash_ptr("mt_chip", (uint32_t)i), cr, labels[i],
                            app->cfg.max_tokens == presets[i])) {
                app->cfg.max_tokens = presets[i];
                dirty = true;
            }
        }
    }

    lfm_rect_cut_top(&inner, 14);
    lfm_ui_draw_text_cstr(ui, LFM_UI_FONT_SMALL, inner.x, inner.y, ui->theme.text_muted, "Context Window");
    lfm_rect_cut_top(&inner, 18);
    {
        static const int32_t ctxs[3] = {2048, 10000, 32768};
        static const char *clabels[3] = {"2k", "10k", "32k"};
        LfmUiRect row = lfm_rect_cut_top(&inner, 26);
        int32_t cw = (row.w - 12) / 3;
        for (int i = 0; i < 3; ++i) {
            LfmUiRect cr = lfm_rect(row.x + i * (cw + 6), row.y, cw, 24);
            if (lfm_ui_chip(ui, lfm_ui_hash_ptr("ctx_chip", (uint32_t)i), cr, clabels[i],
                            app->cfg.ctx_len == ctxs[i])) {
                app->cfg.ctx_len = ctxs[i];
                dirty = true;
            }
        }
    }

    lfm_rect_cut_top(&inner, 12);
    dirty |= lfm_ui_slider_i(ui, lfm_ui_hash_str("set_threads"), lfm_rect_cut_top(&inner, 44),
                             "CPU Threads (0 = Auto)", &app->cfg.n_threads, 0, 16, "");
    lfm_rect_cut_top(&inner, 8);
    dirty |= lfm_ui_toggle(ui, lfm_ui_hash_str("set_warmup"), lfm_rect_cut_top(&inner, 28),
                           "MMAP Page Warmup", &app->cfg.warmup_mmap);

    lfm_rect_cut_top(&inner, 14);
    lfm_ui_draw_text_cstr(ui, LFM_UI_FONT_SMALL, inner.x, inner.y, ui->theme.text_muted, "System Prompt Preset");
    lfm_rect_cut_top(&inner, 18);
    {
        static const char *pnames[3] = {"Helpful", "C/Systems", "Concise"};
        static const char *prompts[3] = {
            "You are a helpful AI assistant.",
            "You are an expert C11 systems engineer. Provide clear, concise, bug-free answers.",
            "Respond concisely and accurately with zero fluff."
        };
        LfmUiRect row = lfm_rect_cut_top(&inner, 26);
        int32_t cw = (row.w - 12) / 3;
        for (int i = 0; i < 3; ++i) {
            LfmUiRect cr = lfm_rect(row.x + i * (cw + 6), row.y, cw, 24);
            bool sel = (strcmp(app->cfg.system_prompt, prompts[i]) == 0);
            if (lfm_ui_chip(ui, lfm_ui_hash_ptr("sys_chip", (uint32_t)i), cr, pnames[i], sel)) {
                snprintf(app->cfg.system_prompt, sizeof(app->cfg.system_prompt), "%s", prompts[i]);
                dirty = true;
            }
        }
    }

    lfm_rect_cut_top(&inner, 16);
    char srv_info[192];
    snprintf(srv_info, sizeof(srv_info), "Server: %s:%d", app->cfg.server_host, app->cfg.server_port);
    lfm_ui_draw_text_cstr(ui, LFM_UI_FONT_SMALL, inner.x, inner.y, ui->theme.text_muted, srv_info);

    if (dirty) {
        pthread_mutex_lock(&app->eng.mu);
        app->eng.cfg = app->cfg;
        pthread_mutex_unlock(&app->eng.mu);
    }
}

static void app_render_welcome(LfmApp *app, LfmUiRect area) {
    LfmUiContext *ui = &app->ui;
    int32_t card_w = area.w > 660 ? 620 : (area.w - 32);
    if (card_w < 260) card_w = 260;
    int32_t card_h = 300;
    LfmUiRect card = lfm_rect(area.x + (area.w - card_w) / 2,
                              area.y + (area.h > card_h + 40 ? (area.h - card_h) / 3 : 20),
                              card_w, card_h);
    lfm_ui_draw_card(ui, card, 14, ui->theme.bg_panel, ui->theme.border);

    LfmUiRect inner = lfm_rect_inset(card, 24, 22);
    lfm_ui_draw_text_cstr(ui, LFM_UI_FONT_TITLE, inner.x, inner.y, ui->theme.text,
                          "Liquid AI — Native C11 Desktop Inference");
    lfm_rect_cut_top(&inner, 30);

    char sub[256];
    snprintf(sub, sizeof(sub), "Model: %.96s   •   Context: %d tokens   •   Default cap: %d tokens",
             app->cfg.model_path[0] ? lfm_gui_basename(app->cfg.model_path) : "None selected",
             app->cfg.ctx_len, app->cfg.max_tokens);
    lfm_ui_draw_text_clipped(ui, LFM_UI_FONT_SMALL, lfm_rect_cut_top(&inner, 22), ui->theme.text_muted, sub);
    lfm_rect_cut_top(&inner, 14);

    static const char *starters[3] = {
        "Explain how Liquid Foundation Models combine ShortConv and GQA layers.",
        "Write a clean, allocation-free C11 SPSC ring buffer with memory barriers.",
        "Compare Q4_K_M quantization vs F16 weights for desktop CPU throughput."
    };
    static const char *btn_labels[3] = {
        "Explain LFM Hybrid Architecture (ShortConv + GQA)",
        "Write an Allocation-Free C11 SPSC Ring Buffer",
        "Compare Q4_K_M vs F16 CPU Inference"
    };

    for (int i = 0; i < 3; ++i) {
        LfmUiRect br = lfm_rect_cut_top(&inner, 38);
        if (lfm_ui_button(ui, lfm_ui_hash_ptr("starter_btn", (uint32_t)i), br, btn_labels[i], false, true)) {
            app_submit_text(app, starters[i]);
        }
        lfm_rect_cut_top(&inner, 8);
    }

    lfm_rect_cut_top(&inner, 8);
    LfmUiRect browse_r = lfm_rect(inner.x, inner.y, 190, 34);
    if (lfm_ui_button(ui, lfm_ui_hash_str("welcome_browse"), browse_r, "Browse .gguf Models...", true, true)) {
        app->show_model_modal = true;
        app_refresh_modal_dir(app, app->modal_dir[0] ? app->modal_dir : "models");
    }
}

static void app_render_chat_pane(LfmApp *app, LfmUiRect area) {
    LfmUiContext *ui = &app->ui;
    lfm_ui_draw_rect(ui, area, ui->theme.bg_app);

    pthread_mutex_lock(&app->eng.mu);
    size_t n_msgs = app->eng.n_msgs;
    pthread_mutex_unlock(&app->eng.mu);

    if (n_msgs == 0) {
        app_render_welcome(app, area);
        return;
    }

    LfmUiRect view = lfm_ui_scroll_begin(ui, lfm_ui_hash_str("chat_scroll"), &app->chat_scroll, area);
    int32_t max_col_w = view.w > 900 ? 860 : (view.w - 32);
    if (max_col_w < 260) max_col_w = 260;
    int32_t col_x = view.x + (view.w - max_col_w) / 2;

    int32_t cur_y = view.y + 16 - app->chat_scroll.offset_y;
    int32_t start_y = cur_y;

    for (size_t i = 0; i < n_msgs; ++i) {
        /* Copy lightweight snapshot of message under mutex so Xft text measurement never holds lock across UI */
        pthread_mutex_lock(&app->eng.mu);
        if (i >= app->eng.n_msgs) {
            pthread_mutex_unlock(&app->eng.mu);
            break;
        }
        LfmGuiMessage *m = &app->eng.msgs[i];
        LfmGuiRole role = m->role;
        bool think_exp = m->think_expanded;
        bool streaming = m->streaming;
        uint32_t gen_tok = m->gen_tokens;
        double dec_tps = m->decode_tok_s;
        char stop_r[64];
        snprintf(stop_r, sizeof(stop_r), "%s", m->stop_reason);

        size_t tlen = m->think.len;
        size_t alen = m->answer.len;
        char *tcopy = tlen ? (char *)malloc(tlen + 1) : NULL;
        char *acopy = alen ? (char *)malloc(alen + 1) : NULL;
        if (tcopy) memcpy(tcopy, m->think.data, tlen + 1);
        if (acopy) memcpy(acopy, m->answer.data, alen + 1);
        pthread_mutex_unlock(&app->eng.mu);

        if (role == LFM_MSG_USER) {
            int32_t max_bubble_w = (max_col_w * 74) / 100;
            int32_t inner_max_w = max_bubble_w - 28;
            int32_t max_lw = 0;
            int32_t th = lfm_ui_measure_wrapped_text(ui, LFM_UI_FONT_UI, acopy ? acopy : "", alen,
                                                     inner_max_w, &max_lw);
            int32_t bw = (max_lw < 80 ? 80 : max_lw) + 28;
            if (bw > max_bubble_w) bw = max_bubble_w;
            int32_t bh = th + 20;
            LfmUiRect box = lfm_rect(col_x + max_col_w - bw, cur_y, bw, bh);
            if (box.y + box.h >= view.y - 40 && box.y <= view.y + view.h + 40) {
                lfm_ui_draw_card(ui, box, 12, ui->theme.bg_user_msg, ui->theme.border);
                lfm_ui_draw_wrapped_text(ui, LFM_UI_FONT_UI, box.x + 14, box.y + 10,
                                         bw - 28, ui->theme.text, acopy ? acopy : "", alen);
            }
            cur_y += bh + 14;
        } else {
            int32_t pad = 16;
            int32_t cw = max_col_w - 2 * pad;
            int32_t think_hdr_h = (tlen > 0) ? 28 : 0;
            bool show_think_body = (tlen > 0) && (think_exp || (streaming && alen == 0));
            int32_t think_body_h = 0;
            if (show_think_body) {
                think_body_h = lfm_ui_measure_wrapped_text(ui, LFM_UI_FONT_SMALL, tcopy ? tcopy : "",
                                                           tlen, cw - 20, NULL) + 20;
            }
            int32_t ans_h = lfm_ui_measure_markdown(ui, acopy ? acopy : "", alen, cw);
            int32_t foot_h = 26;
            int32_t bh = 14 + (think_hdr_h ? think_hdr_h + 6 : 0) +
                         (think_body_h ? think_body_h + 8 : 0) + ans_h + 10 + foot_h + 8;
            LfmUiRect card = lfm_rect(col_x, cur_y, max_col_w, bh);

            bool visible = (card.y + card.h >= view.y - 60 && card.y <= view.y + view.h + 60);
            int32_t y_cursor = card.y + 14;
            if (visible) {
                lfm_ui_draw_card(ui, card, 12, ui->theme.bg_assistant_msg, ui->theme.border);
            }

            if (think_hdr_h > 0) {
                char th_lbl[64];
                snprintf(th_lbl, sizeof(th_lbl), "%s Thinking (%zu chars)",
                         show_think_body ? "▾" : "▸", tlen);
                LfmUiRect tr = lfm_rect(card.x + pad, y_cursor, 210, 26);
                if (visible && lfm_ui_chip(ui, lfm_ui_hash_ptr("think_tog", (uint32_t)i), tr, th_lbl, show_think_body)) {
                    pthread_mutex_lock(&app->eng.mu);
                    if (i < app->eng.n_msgs) {
                        app->eng.msgs[i].think_expanded = !app->eng.msgs[i].think_expanded;
                    }
                    pthread_mutex_unlock(&app->eng.mu);
                }
                y_cursor += think_hdr_h + 6;
                if (show_think_body) {
                    LfmUiRect tbox = lfm_rect(card.x + pad, y_cursor, cw, think_body_h);
                    if (visible) {
                        lfm_ui_draw_card(ui, tbox, 8, ui->theme.bg_think, ui->theme.border);
                        lfm_ui_draw_wrapped_text(ui, LFM_UI_FONT_SMALL, tbox.x + 10, tbox.y + 8,
                                                 cw - 20, ui->theme.text_muted, tcopy ? tcopy : "", tlen);
                    }
                    y_cursor += think_body_h + 8;
                }
            }

            if (visible) {
                lfm_ui_draw_markdown(ui, card.x + pad, y_cursor, cw, acopy ? acopy : "", alen, streaming);
            }
            y_cursor += ans_h + 8;

            /* Footer row: metrics + stop notice + copy button */
            if (visible) {
                LfmUiRect copy_r = lfm_rect(card.x + max_col_w - pad - 62, card.y + bh - 32, 62, 22);
                if (alen > 0 && lfm_ui_chip(ui, lfm_ui_hash_ptr("copy_msg", (uint32_t)i), copy_r, "Copy", false)) {
                    if (ui->backend.clipboard_set && acopy) {
                        ui->backend.clipboard_set(ui->backend.userdata, acopy, alen);
                        snprintf(app->copied_toast, sizeof(app->copied_toast), "Copied");
                        app->copied_until_ms = lfm_gui_now_ms() + 1500;
                    }
                }

                char meta[128];
                if (streaming) {
                    snprintf(meta, sizeof(meta), "streaming  •  %u tok  •  %.1f tok/s", gen_tok, dec_tps);
                } else {
                    snprintf(meta, sizeof(meta), "%u tok  •  %.1f tok/s%s%s",
                             gen_tok, dec_tps,
                             stop_r[0] ? "  •  " : "", stop_r);
                }
                LfmUiColor mc = (strcmp(stop_r, "length") == 0 || strcmp(stop_r, "context_full") == 0)
                                    ? ui->theme.warn : ui->theme.text_muted;
                lfm_ui_draw_text_cstr(ui, LFM_UI_FONT_SMALL, card.x + pad, card.y + bh - 28, mc, meta);
            }
            cur_y += bh + 14;
        }

        free(tcopy);
        free(acopy);
    }

    int32_t total_content_h = (cur_y - start_y) + 16;
    lfm_ui_scroll_end(ui, lfm_ui_hash_str("chat_scroll"), &app->chat_scroll, area, total_content_h);
}

static void app_render_composer(LfmApp *app, LfmUiRect r) {
    LfmUiContext *ui = &app->ui;
    lfm_ui_draw_rect(ui, r, ui->theme.bg_header);
    lfm_ui_draw_rect(ui, lfm_rect(r.x, r.y, r.w, 1), ui->theme.border);

    pthread_mutex_lock(&app->eng.mu);
    bool generating = (app->eng.status == LFM_GUI_GENERATING || app->eng.status == LFM_GUI_LOADING_MODEL);
    uint32_t cpos = app->eng.ctx_pos;
    uint32_t cmax = app->eng.ctx_max;
    double pre_ms = app->eng.last_prefill_ms;
    double dec_tps = app->eng.last_decode_tps;
    double rss = app->eng.rss_mb;
    char err[256];
    snprintf(err, sizeof(err), "%s", app->eng.error_text);
    pthread_mutex_unlock(&app->eng.mu);

    LfmUiRect inner = lfm_rect_inset(r, 14, 10);
    LfmUiRect status_row = lfm_rect_cut_bottom(&inner, 20);
    lfm_rect_cut_bottom(&inner, 6);

    int32_t col_w = inner.w > 900 ? 860 : inner.w;
    int32_t col_x = inner.x + (inner.w - col_w) / 2;
    LfmUiRect comp_row = lfm_rect(col_x, inner.y, col_w, inner.h);

    LfmUiRect send_r = lfm_rect_cut_right(&comp_row, 96);
    lfm_rect_cut_right(&comp_row, 10);

    LfmUiEditResult er = lfm_ui_textedit_widget(ui, lfm_ui_hash_str("composer_edit"), &app->composer,
                                                comp_row, "Message LFM... (Enter to send, Shift+Enter for newline)", true);
    if (!generating && er == LFM_UI_EDIT_SUBMIT && app->composer.len > 0) {
        app_submit_text(app, app->composer.buf);
    }

    if (generating) {
        if (lfm_ui_button(ui, lfm_ui_hash_str("btn_stop"), send_r, "Stop", true, true)) {
            lfm_gui_engine_cancel(&app->eng);
        }
    } else {
        if (lfm_ui_button(ui, lfm_ui_hash_str("btn_send"), send_r, "Send", true, app->composer.len > 0)) {
            app_submit_text(app, app->composer.buf);
        }
    }

    /* Telemetry footer */
    char tele[256];
    if (err[0]) {
        snprintf(tele, sizeof(tele), "Error: %.230s", err);
        lfm_ui_draw_text_clipped(ui, LFM_UI_FONT_SMALL, lfm_rect(col_x, status_row.y + 2, col_w, 18),
                                 ui->theme.danger, tele);
    } else {
        const char *toast = (lfm_gui_now_ms() < app->copied_until_ms) ? "  •  Copied to clipboard" : "";
        snprintf(tele, sizeof(tele),
                 "ctx: %u / %u   •   prefill: %.0f ms   •   decode: %.1f tok/s   •   RSS: %.0f MB   •   max_tokens: %d%s",
                 cpos, cmax, pre_ms, dec_tps, rss, app->cfg.max_tokens, toast);
        lfm_ui_draw_text_clipped(ui, LFM_UI_FONT_SMALL, lfm_rect(col_x, status_row.y + 2, col_w, 18),
                                 ui->theme.text_muted, tele);
    }
}

static void app_render_model_modal(LfmApp *app, LfmUiRect screen_r) {
    if (!app->show_model_modal) return;
    LfmUiContext *ui = &app->ui;

    int32_t mw = screen_r.w > 640 ? 580 : (screen_r.w - 40);
    int32_t mh = screen_r.h > 500 ? 440 : (screen_r.h - 40);
    LfmUiRect modal = lfm_rect((screen_r.w - mw) / 2, (screen_r.h - mh) / 2, mw, mh);

    lfm_ui_draw_card(ui, modal, 12, ui->theme.bg_panel, ui->theme.border_focus);
    LfmUiRect inner = lfm_rect_inset(modal, 18, 16);

    LfmUiRect hdr = lfm_rect_cut_top(&inner, 32);
    LfmUiRect close_r = lfm_rect_cut_right(&hdr, 74);
    lfm_ui_draw_text_cstr(ui, LFM_UI_FONT_BOLD, hdr.x, hdr.y + 6, ui->theme.text, "Select Local .gguf Model");
    if (lfm_ui_button(ui, lfm_ui_hash_str("modal_close"), close_r, "Close", false, true)) {
        app->show_model_modal = false;
        return;
    }

    lfm_rect_cut_top(&inner, 6);
    LfmUiRect dir_bar = lfm_rect_cut_top(&inner, 24);
    lfm_ui_draw_text_clipped(ui, LFM_UI_FONT_SMALL, dir_bar, ui->theme.text_muted, app->modal_dir);
    lfm_rect_cut_top(&inner, 8);

    LfmUiRect list_view = lfm_ui_scroll_begin(ui, lfm_ui_hash_str("modal_list"), &app->modal_scroll, inner);
    int32_t y = list_view.y + 4 - app->modal_scroll.offset_y;
    int32_t start_y = y;

    if (app->modal_count == 0) {
        lfm_ui_draw_text_cstr(ui, LFM_UI_FONT_UI, list_view.x + 8, y + 12, ui->theme.text_muted,
                              "No .gguf files or subdirectories found here.");
        y += 40;
    }

    for (size_t i = 0; i < app->modal_count; ++i) {
        const LfmGuiFileEntry *e = &app->modal_files[i];
        LfmUiRect row = lfm_rect(list_view.x, y, list_view.w, 34);
        y += 38;
        if (row.y + row.h < list_view.y || row.y > list_view.y + list_view.h) continue;

        char label[320];
        if (e->is_dir) {
            snprintf(label, sizeof(label), "[DIR]  %s", e->name);
        } else {
            double mb = (double)e->size_bytes / (1024.0 * 1024.0);
            if (mb >= 1024.0) snprintf(label, sizeof(label), "%s   (%.2f GB)", e->name, mb / 1024.0);
            else snprintf(label, sizeof(label), "%s   (%.1f MB)", e->name, mb);
        }

        bool is_sel = (!e->is_dir && strcmp(e->full_path, app->cfg.model_path) == 0);
        if (lfm_ui_button(ui, lfm_ui_hash_ptr("modal_entry", (uint32_t)i), row, label, is_sel, true)) {
            if (e->is_dir) {
                char next_dir[512];
                snprintf(next_dir, sizeof(next_dir), "%s", e->full_path);
                app_refresh_modal_dir(app, next_dir);
                break;
            } else {
                snprintf(app->cfg.model_path, sizeof(app->cfg.model_path), "%s", e->full_path);
                lfm_gui_engine_load_model(&app->eng, e->full_path);
                app->show_model_modal = false;
                break;
            }
        }
    }

    lfm_ui_scroll_end(ui, lfm_ui_hash_str("modal_list"), &app->modal_scroll, inner, (y - start_y) + 8);
}

static void app_render(LfmApp *app) {
    x11_ensure_backbuf(&app->x11, app->x11.buf_w, app->x11.buf_h);
    lfm_ui_begin_frame(&app->ui, app->x11.buf_w, app->x11.buf_h);

    LfmUiRect root = lfm_rect(0, 0, app->ui.win_w, app->ui.win_h);
    lfm_ui_draw_rect(&app->ui, root, app->ui.theme.bg_app);

    LfmUiRect header = lfm_rect_cut_top(&root, 54);
    app_render_header(app, header);

    LfmUiRect footer = lfm_rect_cut_bottom(&root, 108);
    if (app->show_settings && root.w > 560) {
        LfmUiRect drawer = lfm_rect_cut_right(&root, 300);
        app_render_settings_drawer(app, drawer);
    }

    if (!app->show_model_modal) {
        app_render_chat_pane(app, root);
        app_render_composer(app, footer);
    } else {
        /* Draw dimmed chat behind modal; avoid modal click-through into widgets */
        app_render_model_modal(app, lfm_rect(0, 0, app->ui.win_w, app->ui.win_h));
    }

    lfm_ui_end_frame(&app->ui);
    x11_set_clip(&app->x11, lfm_rect(0, 0, app->x11.buf_w, app->x11.buf_h));
    XCopyArea(app->x11.dpy, app->x11.backbuf, app->x11.win, app->x11.gc,
              0, 0, (unsigned int)app->x11.buf_w, (unsigned int)app->x11.buf_h, 0, 0);
    XFlush(app->x11.dpy);
}

/* ============================================================================
 * 3. CLI & X11 Event Loop
 * ============================================================================ */

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  -m, --model <path.gguf>    Path to local GGUF model (auto-detects ./models/*.gguf)\n"
            "  --ctx <tokens>             Context window size (default: %u)\n"
            "  -n, --max-tokens <tokens>  Max generated tokens per turn, 0=unlimited (default: %u)\n"
            "  -j, --threads <n>          Worker CPU threads (default: 0 = auto)\n"
            "  --server <host[:port]>     Use remote lfm2-server OpenAI SSE backend (e.g. 127.0.0.1:8080)\n"
            "  --warmup                   Enable sequential mmap page warmup on load\n"
            "  --light                    Start with light theme\n"
            "  --smoke-frames <n>         Render N frames cleanly then exit (CI smoke check)\n"
            "  -h, --help                 Show this help message\n",
            prog, LFM_DEFAULT_CTX_LEN, LFM_DEFAULT_MAX_TOKENS);
}

int main(int argc, char **argv) {
    setlocale(LC_ALL, "");
    signal(SIGPIPE, SIG_IGN);

    LfmGuiConfig cfg;
    lfm_gui_config_defaults(&cfg);
    bool dark = true;
    int32_t smoke_frames = 0;

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            snprintf(cfg.model_path, sizeof(cfg.model_path), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            cfg.ctx_len = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc) {
            cfg.max_tokens = lfm_parse_max_tokens_arg(argv[++i]);
        } else if ((strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--threads") == 0) && i + 1 < argc) {
            cfg.n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--server") == 0 && i + 1 < argc) {
            cfg.backend_mode = LFM_BACKEND_SERVER;
            const char *arg = argv[++i];
            const char *colon = strrchr(arg, ':');
            if (colon) {
                size_t hlen = (size_t)(colon - arg);
                if (hlen >= sizeof(cfg.server_host)) hlen = sizeof(cfg.server_host) - 1;
                memcpy(cfg.server_host, arg, hlen);
                cfg.server_host[hlen] = '\0';
                cfg.server_port = atoi(colon + 1);
            } else {
                snprintf(cfg.server_host, sizeof(cfg.server_host), "%s", arg);
            }
        } else if (strcmp(argv[i], "--warmup") == 0) {
            cfg.warmup_mmap = true;
        } else if (strcmp(argv[i], "--light") == 0) {
            dark = false;
        } else if (strcmp(argv[i], "--smoke-frames") == 0 && i + 1 < argc) {
            smoke_frames = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    LfmApp *app = (LfmApp *)calloc(1, sizeof(LfmApp));
    if (!app) return 1;
    app->cfg = cfg;
    app->dark_theme = dark;

    if (!x11_init(&app->x11, 1080, 740)) {
        fprintf(stderr, "Error: cannot connect to X11 display (check DISPLAY env var).\n");
        free(app);
        return 1;
    }

    LfmUiBackend be;
    memset(&be, 0, sizeof(be));
    be.userdata = &app->x11;
    be.draw_rect = x11_draw_rect;
    be.draw_rounded_rect = x11_draw_rounded_rect;
    be.set_clip = x11_set_clip;
    be.measure_text = x11_measure_text;
    be.line_height = x11_line_height;
    be.draw_text = x11_draw_text;
    be.set_cursor = x11_set_cursor;
    be.clipboard_set = x11_clipboard_set;
    be.clipboard_get = x11_clipboard_get;
    be.now_ms = x11_now_ms;

    lfm_ui_init(&app->ui, be, dark);
    lfm_ui_scroll_init(&app->chat_scroll, true);
    lfm_ui_scroll_init(&app->modal_scroll, false);
    lfm_ui_textedit_init(&app->composer, 1024);
    app->ui.focus_id = lfm_ui_hash_str("composer_edit");
    app->composer.focused = true;

    struct stat st;
    if (stat("models", &st) == 0 && S_ISDIR(st.st_mode)) {
        snprintf(app->modal_dir, sizeof(app->modal_dir), "models");
    } else {
        snprintf(app->modal_dir, sizeof(app->modal_dir), ".");
    }

    lfm_gui_engine_init(&app->eng, &app->cfg, x11_wake_callback, &app->x11);
    app_render(app);

    int32_t frames_rendered = 1;
    int x_fd = ConnectionNumber(app->x11.dpy);

    while (!app->x11.should_quit) {
        if (smoke_frames > 0 && frames_rendered >= smoke_frames) break;

        struct pollfd fds[2];
        fds[0].fd = x_fd;
        fds[0].events = POLLIN;
        fds[1].fd = app->x11.wake_pipe[0];
        fds[1].events = POLLIN;

        if (!XPending(app->x11.dpy)) {
            int timeout_ms = (app->copied_until_ms > lfm_gui_now_ms()) ? 100 : -1;
            poll(fds, 2, timeout_ms);
        }

        /* Drain worker wakeup pipe */
        char drain[64];
        while (read(app->x11.wake_pipe[0], drain, sizeof(drain)) > 0) {}

        bool need_draw = true;
        while (XPending(app->x11.dpy)) {
            XEvent ev;
            XNextEvent(app->x11.dpy, &ev);
            if (XFilterEvent(&ev, None)) continue;

            switch (ev.type) {
                case ClientMessage:
                    if ((Atom)ev.xclient.data.l[0] == app->x11.wm_delete) {
                        app->x11.should_quit = true;
                    }
                    break;
                case ConfigureNotify:
                    if (ev.xconfigure.width > 0 && ev.xconfigure.height > 0) {
                        x11_ensure_backbuf(&app->x11, ev.xconfigure.width, ev.xconfigure.height);
                    }
                    break;
                case SelectionRequest:
                    x11_handle_selection_request(&app->x11, &ev.xselectionrequest);
                    break;
                case MotionNotify:
                    app->ui.mouse_x = ev.xmotion.x;
                    app->ui.mouse_y = ev.xmotion.y;
                    break;
                case ButtonPress:
                    app->ui.mouse_x = ev.xbutton.x;
                    app->ui.mouse_y = ev.xbutton.y;
                    if (ev.xbutton.button == Button1) {
                        app->ui.mouse_down = true;
                        app->ui.mouse_pressed = true;
                    } else if (ev.xbutton.button == Button4) {
                        app->ui.wheel_dy += 1;
                    } else if (ev.xbutton.button == Button5) {
                        app->ui.wheel_dy -= 1;
                    }
                    break;
                case ButtonRelease:
                    app->ui.mouse_x = ev.xbutton.x;
                    app->ui.mouse_y = ev.xbutton.y;
                    if (ev.xbutton.button == Button1) {
                        app->ui.mouse_down = false;
                        app->ui.mouse_released = true;
                    }
                    break;
                case KeyPress: {
                    char utf8[32];
                    KeySym ks = NoSymbol;
                    Status status = 0;
                    int n = 0;
                    if (app->x11.xic) {
                        n = Xutf8LookupString(app->x11.xic, &ev.xkey, utf8, sizeof(utf8) - 1, &ks, &status);
                    } else {
                        n = XLookupString(&ev.xkey, utf8, sizeof(utf8) - 1, &ks, NULL);
                    }
                    if (n < 0) n = 0;
                    utf8[n] = '\0';
                    uint32_t mods = 0;
                    if (ev.xkey.state & ShiftMask) mods |= LFM_UI_MOD_SHIFT;
                    if (ev.xkey.state & ControlMask) mods |= LFM_UI_MOD_CTRL;
                    if (ev.xkey.state & Mod1Mask) mods |= LFM_UI_MOD_ALT;
                    lfm_ui_push_key(&app->ui, translate_keysym(ks), mods, utf8, (size_t)n);
                    break;
                }
                default:
                    break;
            }
        }

        if (need_draw && !app->x11.should_quit) {
            app_render(app);
            frames_rendered++;
        }
    }

    lfm_gui_engine_destroy(&app->eng);
    lfm_ui_textedit_free(&app->composer);
    x11_destroy(&app->x11);
    free(app);
    return 0;
}

