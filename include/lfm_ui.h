#ifndef LFM_UI_H
#define LFM_UI_H

/*
 * lfm_ui.h — Single-header C11 immediate-mode GUI library for LFM desktop tools.
 *
 * Define LFM_UI_IMPLEMENTATION in exactly one translation unit before including
 * this header to emit the implementation. All other translation units receive
 * only the public declarations and inline geometry helpers.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t r, g, b, a;
} LfmUiColor;

typedef struct {
    int32_t x, y, w, h;
} LfmUiRect;

typedef enum {
    LFM_UI_FONT_UI = 0,
    LFM_UI_FONT_BOLD = 1,
    LFM_UI_FONT_MONO = 2,
    LFM_UI_FONT_SMALL = 3,
    LFM_UI_FONT_TITLE = 4,
    LFM_UI_FONT_COUNT = 5
} LfmUiFontRole;

typedef enum {
    LFM_UI_CURSOR_ARROW = 0,
    LFM_UI_CURSOR_IBEAM = 1,
    LFM_UI_CURSOR_HAND = 2
} LfmUiCursorKind;

typedef enum {
    LFM_UI_KEY_NONE = 0,
    LFM_UI_KEY_BACKSPACE,
    LFM_UI_KEY_DELETE,
    LFM_UI_KEY_LEFT,
    LFM_UI_KEY_RIGHT,
    LFM_UI_KEY_UP,
    LFM_UI_KEY_DOWN,
    LFM_UI_KEY_HOME,
    LFM_UI_KEY_END,
    LFM_UI_KEY_ENTER,
    LFM_UI_KEY_TAB,
    LFM_UI_KEY_ESCAPE
} LfmUiKey;

enum {
    LFM_UI_MOD_SHIFT = 1 << 0,
    LFM_UI_MOD_CTRL  = 1 << 1,
    LFM_UI_MOD_ALT   = 1 << 2
};

typedef enum {
    LFM_UI_EDIT_NONE = 0,
    LFM_UI_EDIT_CHANGED = 1,
    LFM_UI_EDIT_SUBMIT = 2
} LfmUiEditResult;

typedef struct {
    bool        dark;
    LfmUiColor  bg_app;
    LfmUiColor  bg_header;
    LfmUiColor  bg_panel;
    LfmUiColor  bg_card;
    LfmUiColor  bg_user_msg;
    LfmUiColor  bg_assistant_msg;
    LfmUiColor  bg_think;
    LfmUiColor  bg_code;
    LfmUiColor  bg_input;
    LfmUiColor  bg_hover;
    LfmUiColor  bg_active;
    LfmUiColor  border;
    LfmUiColor  border_focus;
    LfmUiColor  text;
    LfmUiColor  text_muted;
    LfmUiColor  text_code;
    LfmUiColor  accent;
    LfmUiColor  accent_hover;
    LfmUiColor  accent_text;
    LfmUiColor  danger;
    LfmUiColor  success;
    LfmUiColor  warn;
    LfmUiColor  selection;
    LfmUiColor  scrollbar_track;
    LfmUiColor  scrollbar_thumb;
} LfmUiTheme;

typedef struct LfmUiBackend {
    void     *userdata;
    void    (*draw_rect)(void *ud, LfmUiRect r, LfmUiColor c);
    void    (*draw_rounded_rect)(void *ud, LfmUiRect r, int32_t radius,
                                 LfmUiColor fill, LfmUiColor border, int32_t border_w);
    void    (*set_clip)(void *ud, LfmUiRect r);
    int32_t (*measure_text)(void *ud, LfmUiFontRole font, const char *text, size_t len);
    int32_t (*line_height)(void *ud, LfmUiFontRole font);
    void    (*draw_text)(void *ud, LfmUiFontRole font, int32_t x, int32_t y,
                         LfmUiColor c, const char *text, size_t len);
    void    (*set_cursor)(void *ud, LfmUiCursorKind kind);
    void    (*clipboard_set)(void *ud, const char *text, size_t len);
    size_t  (*clipboard_get)(void *ud, char *out, size_t cap);
    uint64_t (*now_ms)(void *ud);
} LfmUiBackend;

typedef struct {
    int32_t offset_y;
    int32_t content_h;
    int32_t view_h;
    bool    sticky_bottom;
    bool    dragging_thumb;
    int32_t drag_grab_y;
} LfmUiScroll;

typedef struct {
    char   *buf;
    size_t  len;
    size_t  cap;
    size_t  cursor;
    size_t  sel_anchor;
    bool    has_sel;
    int32_t scroll_y;
    bool    focused;
    bool    dragging_mouse;
} LfmUiTextEdit;

#define LFM_UI_CLIP_STACK_MAX 16
#define LFM_UI_MAX_KEY_EVENTS 16

typedef struct {
    LfmUiKey key;
    uint32_t mods;
    char     utf8[32];
    size_t   utf8_len;
} LfmUiKeyEvent;

typedef struct {
    LfmUiBackend  backend;
    LfmUiTheme    theme;
    int32_t       win_w;
    int32_t       win_h;

    int32_t       mouse_x;
    int32_t       mouse_y;
    bool          mouse_down;
    bool          mouse_pressed;
    bool          mouse_released;
    int32_t       wheel_dy; /* Positive = scroll up, negative = scroll down */

    uint32_t      hot_id;
    uint32_t      active_id;
    uint32_t      focus_id;
    LfmUiCursorKind wanted_cursor;

    LfmUiRect     clip_stack[LFM_UI_CLIP_STACK_MAX];
    int32_t       clip_depth;

    LfmUiKeyEvent keys[LFM_UI_MAX_KEY_EVENTS];
    int32_t       n_keys;

    bool          needs_redraw;
    uint64_t      frame_ms;
} LfmUiContext;

/* -------- Color & Rect Inline Helpers -------- */

static inline LfmUiColor lfm_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    LfmUiColor c = {r, g, b, a};
    return c;
}

static inline LfmUiColor lfm_rgb(uint8_t r, uint8_t g, uint8_t b) {
    return lfm_rgba(r, g, b, 255);
}

static inline LfmUiRect lfm_rect(int32_t x, int32_t y, int32_t w, int32_t h) {
    LfmUiRect r = {x, y, w < 0 ? 0 : w, h < 0 ? 0 : h};
    return r;
}

static inline LfmUiRect lfm_rect_inset(LfmUiRect r, int32_t dx, int32_t dy) {
    return lfm_rect(r.x + dx, r.y + dy, r.w - 2 * dx, r.h - 2 * dy);
}

static inline LfmUiRect lfm_rect_cut_top(LfmUiRect *r, int32_t h) {
    int32_t take = h > r->h ? r->h : (h < 0 ? 0 : h);
    LfmUiRect out = lfm_rect(r->x, r->y, r->w, take);
    r->y += take;
    r->h -= take;
    return out;
}

static inline LfmUiRect lfm_rect_cut_bottom(LfmUiRect *r, int32_t h) {
    int32_t take = h > r->h ? r->h : (h < 0 ? 0 : h);
    r->h -= take;
    return lfm_rect(r->x, r->y + r->h, r->w, take);
}

static inline LfmUiRect lfm_rect_cut_left(LfmUiRect *r, int32_t w) {
    int32_t take = w > r->w ? r->w : (w < 0 ? 0 : w);
    LfmUiRect out = lfm_rect(r->x, r->y, take, r->h);
    r->x += take;
    r->w -= take;
    return out;
}

static inline LfmUiRect lfm_rect_cut_right(LfmUiRect *r, int32_t w) {
    int32_t take = w > r->w ? r->w : (w < 0 ? 0 : w);
    r->w -= take;
    return lfm_rect(r->x + r->w, r->y, take, r->h);
}

static inline bool lfm_rect_contains(LfmUiRect r, int32_t px, int32_t py) {
    return px >= r.x && px < r.x + r.w && py >= r.y && py < r.y + r.h;
}

static inline LfmUiRect lfm_rect_intersect(LfmUiRect a, LfmUiRect b) {
    int32_t x0 = a.x > b.x ? a.x : b.x;
    int32_t y0 = a.y > b.y ? a.y : b.y;
    int32_t x1 = (a.x + a.w) < (b.x + b.w) ? (a.x + a.w) : (b.x + b.w);
    int32_t y1 = (a.y + a.h) < (b.y + b.h) ? (a.y + a.h) : (b.y + b.h);
    return lfm_rect(x0, y0, x1 - x0, y1 - y0);
}

/* -------- Core Lifecycle & Input -------- */

LfmUiTheme lfm_ui_theme_dark(void);
LfmUiTheme lfm_ui_theme_light(void);

void lfm_ui_init(LfmUiContext *ctx, LfmUiBackend backend, bool dark);
uint32_t lfm_ui_hash_str(const char *s);
uint32_t lfm_ui_hash_ptr(const char *tag, uint32_t index);

void lfm_ui_begin_frame(LfmUiContext *ctx, int32_t w, int32_t h);
void lfm_ui_end_frame(LfmUiContext *ctx);
void lfm_ui_push_key(LfmUiContext *ctx, LfmUiKey key, uint32_t mods, const char *utf8, size_t len);

/* -------- Drawing & Clipping -------- */

void lfm_ui_push_clip(LfmUiContext *ctx, LfmUiRect r);
void lfm_ui_pop_clip(LfmUiContext *ctx);
void lfm_ui_draw_rect(LfmUiContext *ctx, LfmUiRect r, LfmUiColor c);
void lfm_ui_draw_card(LfmUiContext *ctx, LfmUiRect r, int32_t radius, LfmUiColor fill, LfmUiColor border);
int32_t lfm_ui_text_w(LfmUiContext *ctx, LfmUiFontRole font, const char *s, size_t len);
int32_t lfm_ui_line_h(LfmUiContext *ctx, LfmUiFontRole font);
void lfm_ui_draw_text(LfmUiContext *ctx, LfmUiFontRole font, int32_t x, int32_t y, LfmUiColor c, const char *s, size_t len);
void lfm_ui_draw_text_cstr(LfmUiContext *ctx, LfmUiFontRole font, int32_t x, int32_t y, LfmUiColor c, const char *s);
void lfm_ui_draw_text_clipped(LfmUiContext *ctx, LfmUiFontRole font, LfmUiRect r, LfmUiColor c, const char *s);

/* -------- Wrapped Text & Markdown Helpers -------- */

size_t lfm_utf8_next(const char *s, size_t len, size_t pos);
size_t lfm_utf8_prev(const char *s, size_t pos);

int32_t lfm_ui_measure_wrapped_text(LfmUiContext *ctx, LfmUiFontRole font, const char *s, size_t len,
                                    int32_t max_w, int32_t *out_max_line_w);
int32_t lfm_ui_draw_wrapped_text(LfmUiContext *ctx, LfmUiFontRole font, int32_t x, int32_t y,
                                 int32_t max_w, LfmUiColor c, const char *s, size_t len);

int32_t lfm_ui_measure_markdown(LfmUiContext *ctx, const char *s, size_t len, int32_t max_w);
int32_t lfm_ui_draw_markdown(LfmUiContext *ctx, int32_t x, int32_t y, int32_t max_w,
                             const char *s, size_t len, bool show_cursor);

/* -------- Widgets -------- */

bool lfm_ui_button(LfmUiContext *ctx, uint32_t id, LfmUiRect r, const char *label, bool primary, bool enabled);
bool lfm_ui_chip(LfmUiContext *ctx, uint32_t id, LfmUiRect r, const char *label, bool selected);
bool lfm_ui_toggle(LfmUiContext *ctx, uint32_t id, LfmUiRect r, const char *label, bool *val);
bool lfm_ui_slider_f(LfmUiContext *ctx, uint32_t id, LfmUiRect r, const char *label,
                     float *val, float min_v, float max_v, const char *fmt);
bool lfm_ui_slider_i(LfmUiContext *ctx, uint32_t id, LfmUiRect r, const char *label,
                     int32_t *val, int32_t min_v, int32_t max_v, const char *suffix);

/* -------- Scrollable Region -------- */

void lfm_ui_scroll_init(LfmUiScroll *s, bool sticky_bottom);
LfmUiRect lfm_ui_scroll_begin(LfmUiContext *ctx, uint32_t id, LfmUiScroll *s, LfmUiRect r);
void lfm_ui_scroll_end(LfmUiContext *ctx, uint32_t id, LfmUiScroll *s, LfmUiRect r, int32_t content_h);
void lfm_ui_scroll_to_bottom(LfmUiScroll *s);

/* -------- Multi-Line UTF-8 Text Editor -------- */

void lfm_ui_textedit_init(LfmUiTextEdit *ed, size_t initial_cap);
void lfm_ui_textedit_free(LfmUiTextEdit *ed);
void lfm_ui_textedit_clear(LfmUiTextEdit *ed);
void lfm_ui_textedit_set(LfmUiTextEdit *ed, const char *s);
LfmUiEditResult lfm_ui_textedit_widget(LfmUiContext *ctx, uint32_t id, LfmUiTextEdit *ed,
                                       LfmUiRect r, const char *placeholder, bool single_enter_submits);

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * Implementation
 * ============================================================================ */
#ifdef LFM_UI_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LfmUiTheme lfm_ui_theme_dark(void) {
    LfmUiTheme t;
    memset(&t, 0, sizeof(t));
    t.dark             = true;
    t.bg_app           = lfm_rgb(15, 17, 23);
    t.bg_header        = lfm_rgb(21, 24, 33);
    t.bg_panel         = lfm_rgb(24, 28, 38);
    t.bg_card          = lfm_rgb(31, 36, 48);
    t.bg_user_msg      = lfm_rgb(35, 62, 107);
    t.bg_assistant_msg = lfm_rgb(24, 28, 38);
    t.bg_think         = lfm_rgb(22, 26, 35);
    t.bg_code          = lfm_rgb(13, 16, 22);
    t.bg_input         = lfm_rgb(23, 27, 37);
    t.bg_hover         = lfm_rgb(41, 48, 64);
    t.bg_active        = lfm_rgb(52, 60, 80);
    t.border           = lfm_rgb(46, 53, 70);
    t.border_focus     = lfm_rgb(77, 138, 255);
    t.text             = lfm_rgb(235, 239, 247);
    t.text_muted       = lfm_rgb(142, 152, 172);
    t.text_code        = lfm_rgb(174, 214, 255);
    t.accent           = lfm_rgb(66, 133, 244);
    t.accent_hover     = lfm_rgb(90, 150, 252);
    t.accent_text      = lfm_rgb(255, 255, 255);
    t.danger           = lfm_rgb(220, 76, 86);
    t.success          = lfm_rgb(56, 189, 126);
    t.warn             = lfm_rgb(235, 172, 62);
    t.selection        = lfm_rgba(66, 133, 244, 96);
    t.scrollbar_track  = lfm_rgb(20, 23, 31);
    t.scrollbar_thumb  = lfm_rgb(66, 75, 96);
    return t;
}

LfmUiTheme lfm_ui_theme_light(void) {
    LfmUiTheme t;
    memset(&t, 0, sizeof(t));
    t.dark             = false;
    t.bg_app           = lfm_rgb(246, 247, 251);
    t.bg_header        = lfm_rgb(255, 255, 255);
    t.bg_panel         = lfm_rgb(255, 255, 255);
    t.bg_card          = lfm_rgb(238, 241, 247);
    t.bg_user_msg      = lfm_rgb(219, 234, 254);
    t.bg_assistant_msg = lfm_rgb(255, 255, 255);
    t.bg_think         = lfm_rgb(243, 245, 249);
    t.bg_code          = lfm_rgb(236, 239, 246);
    t.bg_input         = lfm_rgb(255, 255, 255);
    t.bg_hover         = lfm_rgb(228, 233, 243);
    t.bg_active        = lfm_rgb(210, 218, 232);
    t.border           = lfm_rgb(214, 220, 232);
    t.border_focus     = lfm_rgb(37, 99, 235);
    t.text             = lfm_rgb(24, 29, 39);
    t.text_muted       = lfm_rgb(100, 110, 130);
    t.text_code        = lfm_rgb(25, 72, 152);
    t.accent           = lfm_rgb(37, 99, 235);
    t.accent_hover     = lfm_rgb(29, 78, 216);
    t.accent_text      = lfm_rgb(255, 255, 255);
    t.danger           = lfm_rgb(205, 52, 65);
    t.success          = lfm_rgb(22, 163, 94);
    t.warn             = lfm_rgb(196, 128, 16);
    t.selection        = lfm_rgba(37, 99, 235, 64);
    t.scrollbar_track  = lfm_rgb(236, 239, 245);
    t.scrollbar_thumb  = lfm_rgb(180, 190, 206);
    return t;
}

void lfm_ui_init(LfmUiContext *ctx, LfmUiBackend backend, bool dark) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->backend = backend;
    ctx->theme = dark ? lfm_ui_theme_dark() : lfm_ui_theme_light();
    ctx->needs_redraw = true;
}

uint32_t lfm_ui_hash_str(const char *s) {
    uint32_t h = 2166136261u;
    if (!s) return h;
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        h ^= *p;
        h *= 16777619u;
    }
    return h ? h : 1u;
}

uint32_t lfm_ui_hash_ptr(const char *tag, uint32_t index) {
    uint32_t h = lfm_ui_hash_str(tag);
    h ^= index + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h ? h : 1u;
}

void lfm_ui_begin_frame(LfmUiContext *ctx, int32_t w, int32_t h) {
    ctx->win_w = w;
    ctx->win_h = h;
    ctx->hot_id = 0;
    ctx->wanted_cursor = LFM_UI_CURSOR_ARROW;
    ctx->clip_depth = 0;
    ctx->clip_stack[0] = lfm_rect(0, 0, w, h);
    if (ctx->backend.set_clip) {
        ctx->backend.set_clip(ctx->backend.userdata, ctx->clip_stack[0]);
    }
    ctx->frame_ms = ctx->backend.now_ms ? ctx->backend.now_ms(ctx->backend.userdata) : 0;
}

void lfm_ui_end_frame(LfmUiContext *ctx) {
    if (ctx->backend.set_cursor) {
        ctx->backend.set_cursor(ctx->backend.userdata, ctx->wanted_cursor);
    }
    if (ctx->mouse_released) {
        ctx->active_id = 0;
    }
    ctx->mouse_pressed = false;
    ctx->mouse_released = false;
    ctx->wheel_dy = 0;
    ctx->n_keys = 0;
}

void lfm_ui_push_key(LfmUiContext *ctx, LfmUiKey key, uint32_t mods, const char *utf8, size_t len) {
    if (ctx->n_keys >= LFM_UI_MAX_KEY_EVENTS) return;
    LfmUiKeyEvent *ev = &ctx->keys[ctx->n_keys++];
    memset(ev, 0, sizeof(*ev));
    ev->key = key;
    ev->mods = mods;
    if (utf8 && len > 0) {
        size_t copy = len < sizeof(ev->utf8) - 1 ? len : sizeof(ev->utf8) - 1;
        memcpy(ev->utf8, utf8, copy);
        ev->utf8_len = copy;
    }
}

void lfm_ui_push_clip(LfmUiContext *ctx, LfmUiRect r) {
    if (ctx->clip_depth + 1 >= LFM_UI_CLIP_STACK_MAX) return;
    LfmUiRect cur = ctx->clip_stack[ctx->clip_depth];
    LfmUiRect next = lfm_rect_intersect(cur, r);
    ctx->clip_stack[++ctx->clip_depth] = next;
    if (ctx->backend.set_clip) {
        ctx->backend.set_clip(ctx->backend.userdata, next);
    }
}

void lfm_ui_pop_clip(LfmUiContext *ctx) {
    if (ctx->clip_depth <= 0) return;
    ctx->clip_depth--;
    if (ctx->backend.set_clip) {
        ctx->backend.set_clip(ctx->backend.userdata, ctx->clip_stack[ctx->clip_depth]);
    }
}

static bool lfm_ui_hit(LfmUiContext *ctx, LfmUiRect r) {
    LfmUiRect clip = ctx->clip_stack[ctx->clip_depth];
    return lfm_rect_contains(lfm_rect_intersect(r, clip), ctx->mouse_x, ctx->mouse_y);
}

void lfm_ui_draw_rect(LfmUiContext *ctx, LfmUiRect r, LfmUiColor c) {
    if (ctx->backend.draw_rect && r.w > 0 && r.h > 0) {
        ctx->backend.draw_rect(ctx->backend.userdata, r, c);
    }
}

void lfm_ui_draw_card(LfmUiContext *ctx, LfmUiRect r, int32_t radius, LfmUiColor fill, LfmUiColor border) {
    if (r.w <= 0 || r.h <= 0) return;
    if (ctx->backend.draw_rounded_rect) {
        ctx->backend.draw_rounded_rect(ctx->backend.userdata, r, radius, fill, border, border.a > 0 ? 1 : 0);
    } else if (ctx->backend.draw_rect) {
        ctx->backend.draw_rect(ctx->backend.userdata, r, fill);
    }
}

int32_t lfm_ui_text_w(LfmUiContext *ctx, LfmUiFontRole font, const char *s, size_t len) {
    if (!ctx->backend.measure_text || !s || len == 0) return 0;
    return ctx->backend.measure_text(ctx->backend.userdata, font, s, len);
}

int32_t lfm_ui_line_h(LfmUiContext *ctx, LfmUiFontRole font) {
    if (!ctx->backend.line_height) return 18;
    int32_t h = ctx->backend.line_height(ctx->backend.userdata, font);
    return h > 0 ? h : 18;
}

void lfm_ui_draw_text(LfmUiContext *ctx, LfmUiFontRole font, int32_t x, int32_t y,
                      LfmUiColor c, const char *s, size_t len) {
    if (!ctx->backend.draw_text || !s || len == 0) return;
    ctx->backend.draw_text(ctx->backend.userdata, font, x, y, c, s, len);
}

void lfm_ui_draw_text_cstr(LfmUiContext *ctx, LfmUiFontRole font, int32_t x, int32_t y,
                           LfmUiColor c, const char *s) {
    if (!s) return;
    lfm_ui_draw_text(ctx, font, x, y, c, s, strlen(s));
}

void lfm_ui_draw_text_clipped(LfmUiContext *ctx, LfmUiFontRole font, LfmUiRect r,
                              LfmUiColor c, const char *s) {
    if (!s || r.w <= 0 || r.h <= 0) return;
    lfm_ui_push_clip(ctx, r);
    int32_t lh = lfm_ui_line_h(ctx, font);
    int32_t ty = r.y + (r.h - lh) / 2;
    lfm_ui_draw_text(ctx, font, r.x, ty, c, s, strlen(s));
    lfm_ui_pop_clip(ctx);
}

/* -------- UTF-8 & Wrapping -------- */

size_t lfm_utf8_next(const char *s, size_t len, size_t pos) {
    if (pos >= len) return len;
    unsigned char b = (unsigned char)s[pos];
    size_t adv = 1;
    if ((b & 0x80u) == 0) adv = 1;
    else if ((b & 0xE0u) == 0xC0u) adv = 2;
    else if ((b & 0xF0u) == 0xE0u) adv = 3;
    else if ((b & 0xF8u) == 0xF0u) adv = 4;
    if (pos + adv > len) return len;
    return pos + adv;
}

size_t lfm_utf8_prev(const char *s, size_t pos) {
    if (pos == 0) return 0;
    size_t p = pos - 1;
    while (p > 0 && (((unsigned char)s[p] & 0xC0u) == 0x80u)) {
        p--;
    }
    return p;
}

typedef struct {
    size_t start;
    size_t end; /* Exclusive, excludes trailing \n */
} LfmUiWrapLine;

static size_t lfm_ui_next_wrap_line(LfmUiContext *ctx, LfmUiFontRole font,
                                    const char *s, size_t len, size_t start,
                                    int32_t max_w, LfmUiWrapLine *out) {
    if (max_w < 24) max_w = 24;
    size_t p = start;
    size_t last_space = (size_t)-1;

    while (p < len) {
        if (s[p] == '\n') {
            out->start = start;
            out->end = p;
            return p + 1;
        }
        size_t next = lfm_utf8_next(s, len, p);
        int32_t w = lfm_ui_text_w(ctx, font, s + start, next - start);
        if (w > max_w && p > start) {
            if (last_space != (size_t)-1 && last_space > start) {
                out->start = start;
                out->end = last_space;
                size_t resume = last_space;
                while (resume < len && (s[resume] == ' ' || s[resume] == '\t')) {
                    resume++;
                }
                return resume;
            }
            out->start = start;
            out->end = p;
            return p;
        }
        if (s[p] == ' ' || s[p] == '\t') {
            last_space = p;
        }
        p = next;
    }

    out->start = start;
    out->end = len;
    return len;
}

static int32_t lfm_ui_wrapped_pass(LfmUiContext *ctx, LfmUiFontRole font,
                                   int32_t x, int32_t y, int32_t max_w,
                                   LfmUiColor c, const char *s, size_t len,
                                   bool do_draw, int32_t *out_max_line_w) {
    int32_t lh = lfm_ui_line_h(ctx, font) + 3;
    if (!s || len == 0) {
        if (out_max_line_w) *out_max_line_w = 0;
        return lh;
    }

    int32_t cur_y = y;
    int32_t max_lw = 0;
    size_t pos = 0;

    while (pos < len) {
        LfmUiWrapLine line;
        size_t next = lfm_ui_next_wrap_line(ctx, font, s, len, pos, max_w, &line);
        size_t line_len = line.end - line.start;
        if (line_len > 0) {
            int32_t lw = lfm_ui_text_w(ctx, font, s + line.start, line_len);
            if (lw > max_lw) max_lw = lw;
            if (do_draw) {
                lfm_ui_draw_text(ctx, font, x, cur_y, c, s + line.start, line_len);
            }
        }
        cur_y += lh;
        if (next == len && line.end < len && s[line.end] == '\n') {
            cur_y += lh;
            break;
        }
        pos = next;
    }

    if (out_max_line_w) *out_max_line_w = max_lw;
    return cur_y - y;
}

int32_t lfm_ui_measure_wrapped_text(LfmUiContext *ctx, LfmUiFontRole font,
                                    const char *s, size_t len,
                                    int32_t max_w, int32_t *out_max_line_w) {
    return lfm_ui_wrapped_pass(ctx, font, 0, 0, max_w, ctx->theme.text, s, len, false, out_max_line_w);
}

int32_t lfm_ui_draw_wrapped_text(LfmUiContext *ctx, LfmUiFontRole font,
                                 int32_t x, int32_t y, int32_t max_w,
                                 LfmUiColor c, const char *s, size_t len) {
    return lfm_ui_wrapped_pass(ctx, font, x, y, max_w, c, s, len, true, NULL);
}

/* -------- Markdown-Lite Blocks (Prose + Fenced Code) -------- */

static int32_t lfm_ui_markdown_pass(LfmUiContext *ctx, int32_t x, int32_t y, int32_t max_w,
                                    const char *s, size_t len, bool do_draw, bool show_cursor) {
    if (!s || len == 0) {
        int32_t lh = lfm_ui_line_h(ctx, LFM_UI_FONT_UI) + 3;
        if (do_draw && show_cursor) {
            lfm_ui_draw_rect(ctx, lfm_rect(x, y + 2, 7, lh - 5), ctx->theme.accent);
        }
        return lh;
    }

    int32_t cur_y = y;
    size_t pos = 0;
    const int32_t pad = 10;

    while (pos < len) {
        /* Locate next fenced code marker at start of line */
        size_t search = pos;
        size_t fence_start = (size_t)-1;
        while (search < len) {
            bool at_bol = (search == pos) || (search > 0 && s[search - 1] == '\n');
            if (at_bol && search + 3 <= len && memcmp(s + search, "```", 3) == 0) {
                fence_start = search;
                break;
            }
            search++;
        }

        size_t prose_end = (fence_start != (size_t)-1) ? fence_start : len;
        if (prose_end > pos) {
            size_t trimmed_end = prose_end;
            while (trimmed_end > pos && s[trimmed_end - 1] == '\n') trimmed_end--;
            if (trimmed_end > pos) {
                int32_t ph = lfm_ui_wrapped_pass(ctx, LFM_UI_FONT_UI, x, cur_y, max_w,
                                                 ctx->theme.text, s + pos, trimmed_end - pos,
                                                 do_draw, NULL);
                cur_y += ph + 4;
            }
        }

        if (fence_start == (size_t)-1) {
            break;
        }

        /* Extract language tag + body until closing ``` or EOF */
        size_t lang_start = fence_start + 3;
        size_t lang_end = lang_start;
        while (lang_end < len && s[lang_end] != '\n') lang_end++;
        size_t code_start = (lang_end < len && s[lang_end] == '\n') ? lang_end + 1 : lang_end;

        size_t close_pos = code_start;
        size_t code_end = len;
        size_t resume = len;
        while (close_pos + 3 <= len) {
            bool at_bol = (close_pos == code_start) || (close_pos > 0 && s[close_pos - 1] == '\n');
            if (at_bol && memcmp(s + close_pos, "```", 3) == 0) {
                code_end = close_pos;
                while (code_end > code_start && s[code_end - 1] == '\n') code_end--;
                resume = close_pos + 3;
                while (resume < len && s[resume] != '\n') resume++;
                if (resume < len && s[resume] == '\n') resume++;
                break;
            }
            close_pos++;
        }
        if (code_end == len) {
            while (code_end > code_start && s[code_end - 1] == '\n') code_end--;
        }

        int32_t inner_w = max_w - 2 * pad;
        if (inner_w < 40) inner_w = 40;
        int32_t header_h = (lang_end > lang_start) ? (lfm_ui_line_h(ctx, LFM_UI_FONT_SMALL) + 6) : 0;
        int32_t code_h = lfm_ui_wrapped_pass(ctx, LFM_UI_FONT_MONO, 0, 0, inner_w,
                                             ctx->theme.text_code, s + code_start,
                                             code_end - code_start, false, NULL);
        int32_t box_h = pad + header_h + code_h + pad;
        if (do_draw) {
            LfmUiRect box = lfm_rect(x, cur_y, max_w, box_h);
            lfm_ui_draw_card(ctx, box, 8, ctx->theme.bg_code, ctx->theme.border);
            int32_t ty = cur_y + pad;
            if (header_h > 0) {
                lfm_ui_draw_text(ctx, LFM_UI_FONT_SMALL, x + pad, ty,
                                 ctx->theme.text_muted, s + lang_start, lang_end - lang_start);
                ty += header_h;
            }
            lfm_ui_wrapped_pass(ctx, LFM_UI_FONT_MONO, x + pad, ty, inner_w,
                                ctx->theme.text_code, s + code_start,
                                code_end - code_start, true, NULL);
        }
        cur_y += box_h + 6;
        pos = resume;
    }

    if (do_draw && show_cursor) {
        int32_t ch = lfm_ui_line_h(ctx, LFM_UI_FONT_UI);
        lfm_ui_draw_rect(ctx, lfm_rect(x, cur_y - ch + 2, 7, ch - 4), ctx->theme.accent);
    }
    return cur_y - y;
}

int32_t lfm_ui_measure_markdown(LfmUiContext *ctx, const char *s, size_t len, int32_t max_w) {
    return lfm_ui_markdown_pass(ctx, 0, 0, max_w, s, len, false, false);
}

int32_t lfm_ui_draw_markdown(LfmUiContext *ctx, int32_t x, int32_t y, int32_t max_w,
                             const char *s, size_t len, bool show_cursor) {
    return lfm_ui_markdown_pass(ctx, x, y, max_w, s, len, true, show_cursor);
}

/* -------- Standard Widgets -------- */

static bool lfm_ui_interactive_state(LfmUiContext *ctx, uint32_t id, LfmUiRect r, bool enabled,
                                     bool *out_hovered, bool *out_pressed_down) {
    bool hovered = enabled && lfm_ui_hit(ctx, r);
    if (hovered) {
        ctx->hot_id = id;
        ctx->wanted_cursor = LFM_UI_CURSOR_HAND;
        if (ctx->mouse_pressed) {
            ctx->active_id = id;
        }
    }
    bool clicked = enabled && hovered && ctx->mouse_released && ctx->active_id == id;
    if (out_hovered) *out_hovered = hovered;
    if (out_pressed_down) *out_pressed_down = (enabled && hovered && ctx->active_id == id && ctx->mouse_down);
    return clicked;
}

bool lfm_ui_button(LfmUiContext *ctx, uint32_t id, LfmUiRect r, const char *label, bool primary, bool enabled) {
    bool hovered = false, held = false;
    bool clicked = lfm_ui_interactive_state(ctx, id, r, enabled, &hovered, &held);

    LfmUiColor fill = primary ? ctx->theme.accent : ctx->theme.bg_card;
    LfmUiColor border = primary ? ctx->theme.accent : ctx->theme.border;
    LfmUiColor fg = primary ? ctx->theme.accent_text : ctx->theme.text;

    if (!enabled) {
        fill = ctx->theme.bg_panel;
        border = ctx->theme.border;
        fg = ctx->theme.text_muted;
    } else if (held) {
        fill = primary ? ctx->theme.accent : ctx->theme.bg_active;
    } else if (hovered) {
        fill = primary ? ctx->theme.accent_hover : ctx->theme.bg_hover;
    }

    lfm_ui_draw_card(ctx, r, 7, fill, border);
    if (label && label[0]) {
        size_t len = strlen(label);
        int32_t tw = lfm_ui_text_w(ctx, LFM_UI_FONT_BOLD, label, len);
        int32_t lh = lfm_ui_line_h(ctx, LFM_UI_FONT_BOLD);
        int32_t tx = r.x + (r.w - tw) / 2;
        if (tx < r.x + 6) tx = r.x + 6;
        int32_t ty = r.y + (r.h - lh) / 2;
        lfm_ui_push_clip(ctx, lfm_rect_inset(r, 4, 2));
        lfm_ui_draw_text(ctx, LFM_UI_FONT_BOLD, tx, ty, fg, label, len);
        lfm_ui_pop_clip(ctx);
    }
    return clicked;
}

bool lfm_ui_chip(LfmUiContext *ctx, uint32_t id, LfmUiRect r, const char *label, bool selected) {
    bool hovered = false, held = false;
    bool clicked = lfm_ui_interactive_state(ctx, id, r, true, &hovered, &held);

    LfmUiColor fill = selected ? ctx->theme.accent : (hovered ? ctx->theme.bg_hover : ctx->theme.bg_card);
    LfmUiColor border = selected ? ctx->theme.accent : ctx->theme.border;
    LfmUiColor fg = selected ? ctx->theme.accent_text : ctx->theme.text;
    lfm_ui_draw_card(ctx, r, r.h / 2, fill, border);

    if (label && label[0]) {
        size_t len = strlen(label);
        int32_t tw = lfm_ui_text_w(ctx, LFM_UI_FONT_SMALL, label, len);
        int32_t lh = lfm_ui_line_h(ctx, LFM_UI_FONT_SMALL);
        lfm_ui_push_clip(ctx, lfm_rect_inset(r, 6, 1));
        lfm_ui_draw_text(ctx, LFM_UI_FONT_SMALL, r.x + (r.w - tw) / 2, r.y + (r.h - lh) / 2, fg, label, len);
        lfm_ui_pop_clip(ctx);
    }
    return clicked;
}

bool lfm_ui_toggle(LfmUiContext *ctx, uint32_t id, LfmUiRect r, const char *label, bool *val) {
    bool hovered = false, held = false;
    bool clicked = lfm_ui_interactive_state(ctx, id, r, true, &hovered, &held);
    if (clicked && val) *val = !(*val);

    bool on = val ? *val : false;
    int32_t pill_w = 38, pill_h = 20;
    LfmUiRect pill = lfm_rect(r.x + r.w - pill_w, r.y + (r.h - pill_h) / 2, pill_w, pill_h);
    LfmUiColor track = on ? ctx->theme.accent : (hovered ? ctx->theme.bg_hover : ctx->theme.bg_card);
    lfm_ui_draw_card(ctx, pill, pill_h / 2, track, on ? ctx->theme.accent : ctx->theme.border);

    int32_t knob = 14;
    int32_t kx = on ? (pill.x + pill_w - knob - 3) : (pill.x + 3);
    int32_t ky = pill.y + (pill_h - knob) / 2;
    lfm_ui_draw_card(ctx, lfm_rect(kx, ky, knob, knob), knob / 2, ctx->theme.accent_text, lfm_rgba(0, 0, 0, 0));

    if (label) {
        LfmUiRect lr = lfm_rect(r.x, r.y, r.w - pill_w - 8, r.h);
        lfm_ui_draw_text_clipped(ctx, LFM_UI_FONT_UI, lr, ctx->theme.text, label);
    }
    return clicked;
}

static bool lfm_ui_slider_norm(LfmUiContext *ctx, uint32_t id, LfmUiRect r, const char *label,
                               const char *val_str, float *norm_t) {
    int32_t top_h = lfm_ui_line_h(ctx, LFM_UI_FONT_SMALL);
    LfmUiRect track_hit = lfm_rect(r.x, r.y + top_h + 2, r.w, r.h - top_h - 2);
    if (track_hit.h < 16) track_hit = r;

    bool hovered = lfm_ui_hit(ctx, track_hit);
    if (hovered) {
        ctx->hot_id = id;
        ctx->wanted_cursor = LFM_UI_CURSOR_HAND;
        if (ctx->mouse_pressed) ctx->active_id = id;
    }

    bool changed = false;
    if (ctx->active_id == id && ctx->mouse_down) {
        float t = (float)(ctx->mouse_x - r.x) / (float)(r.w > 1 ? r.w : 1);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        if (t != *norm_t) {
            *norm_t = t;
            changed = true;
        }
    }

    if (label) {
        lfm_ui_draw_text_cstr(ctx, LFM_UI_FONT_SMALL, r.x, r.y, ctx->theme.text_muted, label);
    }
    if (val_str) {
        size_t vlen = strlen(val_str);
        int32_t vw = lfm_ui_text_w(ctx, LFM_UI_FONT_SMALL, val_str, vlen);
        lfm_ui_draw_text(ctx, LFM_UI_FONT_SMALL, r.x + r.w - vw, r.y, ctx->theme.text, val_str, vlen);
    }

    int32_t bar_y = track_hit.y + track_hit.h / 2 - 3;
    LfmUiRect bar = lfm_rect(r.x, bar_y, r.w, 6);
    lfm_ui_draw_card(ctx, bar, 3, ctx->theme.bg_code, ctx->theme.border);
    int32_t fill_w = (int32_t)((*norm_t) * (float)r.w + 0.5f);
    if (fill_w > 0) {
        lfm_ui_draw_card(ctx, lfm_rect(r.x, bar_y, fill_w, 6), 3, ctx->theme.accent, lfm_rgba(0, 0, 0, 0));
    }
    int32_t kx = r.x + fill_w - 6;
    if (kx < r.x) kx = r.x;
    if (kx > r.x + r.w - 12) kx = r.x + r.w - 12;
    lfm_ui_draw_card(ctx, lfm_rect(kx, bar_y - 3, 12, 12), 6,
                     (hovered || ctx->active_id == id) ? ctx->theme.accent_hover : ctx->theme.accent_text,
                     ctx->theme.accent);
    return changed;
}

bool lfm_ui_slider_f(LfmUiContext *ctx, uint32_t id, LfmUiRect r, const char *label,
                     float *val, float min_v, float max_v, const char *fmt) {
    float span = (max_v > min_v) ? (max_v - min_v) : 1.0f;
    float t = (*val - min_v) / span;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    char buf[48];
    snprintf(buf, sizeof(buf), fmt ? fmt : "%.2f", *val);
    if (lfm_ui_slider_norm(ctx, id, r, label, buf, &t)) {
        *val = min_v + t * span;
        return true;
    }
    return false;
}

bool lfm_ui_slider_i(LfmUiContext *ctx, uint32_t id, LfmUiRect r, const char *label,
                     int32_t *val, int32_t min_v, int32_t max_v, const char *suffix) {
    int32_t span = (max_v > min_v) ? (max_v - min_v) : 1;
    float t = (float)(*val - min_v) / (float)span;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    char buf[48];
    snprintf(buf, sizeof(buf), "%d%s", *val, suffix ? suffix : "");
    if (lfm_ui_slider_norm(ctx, id, r, label, buf, &t)) {
        *val = min_v + (int32_t)(t * (float)span + 0.5f);
        return true;
    }
    return false;
}

/* -------- Scrollable Region -------- */

void lfm_ui_scroll_init(LfmUiScroll *s, bool sticky_bottom) {
    memset(s, 0, sizeof(*s));
    s->sticky_bottom = sticky_bottom;
}

void lfm_ui_scroll_to_bottom(LfmUiScroll *s) {
    s->sticky_bottom = true;
    int32_t max_off = s->content_h - s->view_h;
    s->offset_y = max_off > 0 ? max_off : 0;
}

LfmUiRect lfm_ui_scroll_begin(LfmUiContext *ctx, uint32_t id, LfmUiScroll *s, LfmUiRect r) {
    (void)id;
    s->view_h = r.h > 0 ? r.h : 1;
    if (lfm_ui_hit(ctx, r) && ctx->wheel_dy != 0) {
        s->offset_y -= ctx->wheel_dy * 42;
        int32_t max_off = s->content_h - s->view_h;
        if (max_off < 0) max_off = 0;
        if (s->offset_y < 0) s->offset_y = 0;
        if (s->offset_y > max_off) s->offset_y = max_off;
        s->sticky_bottom = (max_off == 0) || (s->offset_y >= max_off - 16);
        ctx->wheel_dy = 0;
    }
    lfm_ui_push_clip(ctx, r);
    return lfm_rect(r.x, r.y, r.w > 12 ? r.w - 10 : r.w, r.h);
}

void lfm_ui_scroll_end(LfmUiContext *ctx, uint32_t id, LfmUiScroll *s, LfmUiRect r, int32_t content_h) {
    lfm_ui_pop_clip(ctx);
    s->content_h = content_h;
    int32_t max_off = s->content_h - r.h;
    if (max_off < 0) max_off = 0;

    if (s->sticky_bottom) {
        s->offset_y = max_off;
    } else if (s->offset_y > max_off) {
        s->offset_y = max_off;
    } else if (s->offset_y < 0) {
        s->offset_y = 0;
    }

    if (max_off <= 0 || r.w < 16 || r.h < 24) return;

    LfmUiRect track = lfm_rect(r.x + r.w - 8, r.y + 4, 6, r.h - 8);
    int32_t thumb_h = (int32_t)(((double)r.h / (double)s->content_h) * (double)track.h);
    if (thumb_h < 28) thumb_h = 28;
    if (thumb_h > track.h) thumb_h = track.h;

    int32_t travel = track.h - thumb_h;
    int32_t thumb_y = track.y + (travel > 0 ? (int32_t)(((double)s->offset_y / (double)max_off) * (double)travel) : 0);
    LfmUiRect thumb = lfm_rect(track.x, thumb_y, track.w, thumb_h);

    if (lfm_ui_hit(ctx, track)) {
        ctx->hot_id = id;
        ctx->wanted_cursor = LFM_UI_CURSOR_HAND;
        if (ctx->mouse_pressed) {
            ctx->active_id = id;
            s->dragging_thumb = true;
            s->drag_grab_y = lfm_rect_contains(thumb, ctx->mouse_x, ctx->mouse_y)
                                 ? (ctx->mouse_y - thumb.y)
                                 : (thumb_h / 2);
        }
    }
    if (s->dragging_thumb) {
        if (ctx->mouse_down) {
            int32_t rel = ctx->mouse_y - track.y - s->drag_grab_y;
            if (rel < 0) rel = 0;
            if (rel > travel) rel = travel;
            s->offset_y = travel > 0 ? (int32_t)(((double)rel / (double)travel) * (double)max_off) : 0;
            s->sticky_bottom = (s->offset_y >= max_off - 8);
        } else {
            s->dragging_thumb = false;
        }
    }

    lfm_ui_draw_card(ctx, track, 3, ctx->theme.scrollbar_track, lfm_rgba(0, 0, 0, 0));
    lfm_ui_draw_card(ctx, thumb, 3,
                     (s->dragging_thumb || ctx->hot_id == id) ? ctx->theme.accent : ctx->theme.scrollbar_thumb,
                     lfm_rgba(0, 0, 0, 0));
}

/* -------- Multi-Line UTF-8 TextEdit -------- */

void lfm_ui_textedit_init(LfmUiTextEdit *ed, size_t initial_cap) {
    memset(ed, 0, sizeof(*ed));
    ed->cap = initial_cap < 256 ? 256 : initial_cap;
    ed->buf = (char *)calloc(ed->cap, 1);
}

void lfm_ui_textedit_free(LfmUiTextEdit *ed) {
    free(ed->buf);
    memset(ed, 0, sizeof(*ed));
}

void lfm_ui_textedit_clear(LfmUiTextEdit *ed) {
    ed->len = 0;
    ed->cursor = 0;
    ed->sel_anchor = 0;
    ed->has_sel = false;
    ed->scroll_y = 0;
    if (ed->buf) ed->buf[0] = '\0';
}

void lfm_ui_textedit_set(LfmUiTextEdit *ed, const char *s) {
    if (!s) {
        lfm_ui_textedit_clear(ed);
        return;
    }
    size_t n = strlen(s);
    if (n + 1 > ed->cap) {
        size_t next_cap = n + 256;
        char *nb = (char *)realloc(ed->buf, next_cap);
        if (!nb) return;
        ed->buf = nb;
        ed->cap = next_cap;
    }
    memcpy(ed->buf, s, n + 1);
    ed->len = n;
    ed->cursor = n;
    ed->sel_anchor = n;
    ed->has_sel = false;
}

static bool lfm_ui_textedit_ensure(LfmUiTextEdit *ed, size_t extra) {
    if (ed->len + extra + 1 <= ed->cap) return true;
    size_t next_cap = ed->cap * 2 + extra + 64;
    char *nb = (char *)realloc(ed->buf, next_cap);
    if (!nb) return false;
    ed->buf = nb;
    ed->cap = next_cap;
    return true;
}

static void lfm_ui_textedit_sel_range(const LfmUiTextEdit *ed, size_t *a, size_t *b) {
    if (!ed->has_sel || ed->sel_anchor == ed->cursor) {
        *a = *b = ed->cursor;
    } else if (ed->sel_anchor < ed->cursor) {
        *a = ed->sel_anchor;
        *b = ed->cursor;
    } else {
        *a = ed->cursor;
        *b = ed->sel_anchor;
    }
}

static void lfm_ui_textedit_delete_sel(LfmUiTextEdit *ed) {
    size_t a, b;
    lfm_ui_textedit_sel_range(ed, &a, &b);
    if (b > a) {
        memmove(ed->buf + a, ed->buf + b, ed->len - b + 1);
        ed->len -= (b - a);
        ed->cursor = a;
    }
    ed->has_sel = false;
    ed->sel_anchor = ed->cursor;
}

static void lfm_ui_textedit_insert(LfmUiTextEdit *ed, const char *s, size_t slen) {
    if (!s || slen == 0) return;
    if (ed->has_sel) lfm_ui_textedit_delete_sel(ed);
    if (!lfm_ui_textedit_ensure(ed, slen)) return;
    memmove(ed->buf + ed->cursor + slen, ed->buf + ed->cursor, ed->len - ed->cursor + 1);
    memcpy(ed->buf + ed->cursor, s, slen);
    ed->len += slen;
    ed->cursor += slen;
    ed->sel_anchor = ed->cursor;
}

static void lfm_ui_textedit_move(LfmUiTextEdit *ed, size_t next_pos, bool keep_sel) {
    if (keep_sel) {
        if (!ed->has_sel) ed->sel_anchor = ed->cursor;
        ed->cursor = next_pos;
        ed->has_sel = (ed->cursor != ed->sel_anchor);
    } else {
        ed->cursor = next_pos;
        ed->sel_anchor = next_pos;
        ed->has_sel = false;
    }
}

static size_t lfm_ui_textedit_hit_pos(LfmUiContext *ctx, const LfmUiTextEdit *ed,
                                      int32_t rel_x, int32_t rel_y, int32_t wrap_w) {
    int32_t lh = lfm_ui_line_h(ctx, LFM_UI_FONT_UI) + 3;
    int32_t target_row = (rel_y < 0) ? 0 : (rel_y / lh);
    size_t pos = 0;
    int32_t row = 0;

    if (ed->len == 0) return 0;

    while (pos <= ed->len) {
        LfmUiWrapLine line;
        size_t next = lfm_ui_next_wrap_line(ctx, LFM_UI_FONT_UI, ed->buf, ed->len, pos, wrap_w, &line);
        bool is_last = (next >= ed->len) && !(line.end < ed->len && ed->buf[line.end] == '\n');
        if (row == target_row || is_last) {
            size_t best = line.start;
            int32_t best_dist = rel_x < 0 ? -rel_x : rel_x;
            size_t p = line.start;
            while (p < line.end) {
                size_t np = lfm_utf8_next(ed->buf, line.end, p);
                int32_t xw = lfm_ui_text_w(ctx, LFM_UI_FONT_UI, ed->buf + line.start, np - line.start);
                int32_t dist = rel_x > xw ? (rel_x - xw) : (xw - rel_x);
                if (dist <= best_dist) {
                    best_dist = dist;
                    best = np;
                }
                p = np;
            }
            return best;
        }
        pos = next;
        row++;
        if (pos == ed->len && line.end < ed->len && ed->buf[line.end] == '\n') {
            if (row == target_row) return ed->len;
            break;
        }
    }
    return ed->len;
}

LfmUiEditResult lfm_ui_textedit_widget(LfmUiContext *ctx, uint32_t id, LfmUiTextEdit *ed,
                                       LfmUiRect r, const char *placeholder, bool single_enter_submits) {
    LfmUiEditResult res = LFM_UI_EDIT_NONE;
    const int32_t pad = 10;
    LfmUiRect inner = lfm_rect_inset(r, pad, 8);
    if (inner.w < 24) inner.w = 24;
    if (inner.h < 18) inner.h = 18;

    bool hovered = lfm_ui_hit(ctx, r);
    if (hovered) {
        ctx->hot_id = id;
        ctx->wanted_cursor = LFM_UI_CURSOR_IBEAM;
        if (ctx->wheel_dy != 0) {
            ed->scroll_y -= ctx->wheel_dy * 24;
            if (ed->scroll_y < 0) ed->scroll_y = 0;
            ctx->wheel_dy = 0;
        }
        if (ctx->mouse_pressed) {
            ctx->focus_id = id;
            ctx->active_id = id;
            ed->focused = true;
            ed->dragging_mouse = true;
            size_t p = lfm_ui_textedit_hit_pos(ctx, ed, ctx->mouse_x - inner.x,
                                               ctx->mouse_y - inner.y + ed->scroll_y, inner.w);
            lfm_ui_textedit_move(ed, p, false);
        }
    } else if (ctx->mouse_pressed) {
        if (ctx->focus_id == id) ctx->focus_id = 0;
        ed->focused = false;
    }

    if (ed->dragging_mouse) {
        if (ctx->mouse_down) {
            size_t p = lfm_ui_textedit_hit_pos(ctx, ed, ctx->mouse_x - inner.x,
                                               ctx->mouse_y - inner.y + ed->scroll_y, inner.w);
            lfm_ui_textedit_move(ed, p, true);
        } else {
            ed->dragging_mouse = false;
        }
    }

    ed->focused = (ctx->focus_id == id);

    if (ed->focused) {
        for (int32_t i = 0; i < ctx->n_keys; ++i) {
            LfmUiKeyEvent *ev = &ctx->keys[i];
            bool ctrl = (ev->mods & LFM_UI_MOD_CTRL) != 0;
            bool shift = (ev->mods & LFM_UI_MOD_SHIFT) != 0;

            if (ev->key == LFM_UI_KEY_ENTER) {
                if (single_enter_submits && !shift) {
                    res = LFM_UI_EDIT_SUBMIT;
                } else {
                    lfm_ui_textedit_insert(ed, "\n", 1);
                    res = LFM_UI_EDIT_CHANGED;
                }
                continue;
            }
            if (ev->key == LFM_UI_KEY_BACKSPACE) {
                if (ed->has_sel) {
                    lfm_ui_textedit_delete_sel(ed);
                    res = LFM_UI_EDIT_CHANGED;
                } else if (ed->cursor > 0) {
                    size_t prev = lfm_utf8_prev(ed->buf, ed->cursor);
                    memmove(ed->buf + prev, ed->buf + ed->cursor, ed->len - ed->cursor + 1);
                    ed->len -= (ed->cursor - prev);
                    ed->cursor = prev;
                    ed->sel_anchor = prev;
                    res = LFM_UI_EDIT_CHANGED;
                }
                continue;
            }
            if (ev->key == LFM_UI_KEY_DELETE) {
                if (ed->has_sel) {
                    lfm_ui_textedit_delete_sel(ed);
                    res = LFM_UI_EDIT_CHANGED;
                } else if (ed->cursor < ed->len) {
                    size_t next = lfm_utf8_next(ed->buf, ed->len, ed->cursor);
                    memmove(ed->buf + ed->cursor, ed->buf + next, ed->len - next + 1);
                    ed->len -= (next - ed->cursor);
                    res = LFM_UI_EDIT_CHANGED;
                }
                continue;
            }
            if (ev->key == LFM_UI_KEY_LEFT) {
                size_t prev = lfm_utf8_prev(ed->buf, ed->cursor);
                lfm_ui_textedit_move(ed, prev, shift);
                continue;
            }
            if (ev->key == LFM_UI_KEY_RIGHT) {
                size_t next = lfm_utf8_next(ed->buf, ed->len, ed->cursor);
                lfm_ui_textedit_move(ed, next, shift);
                continue;
            }
            if (ev->key == LFM_UI_KEY_HOME) {
                size_t sol = ed->cursor;
                while (sol > 0 && ed->buf[sol - 1] != '\n') sol--;
                lfm_ui_textedit_move(ed, sol, shift);
                continue;
            }
            if (ev->key == LFM_UI_KEY_END) {
                size_t eol = ed->cursor;
                while (eol < ed->len && ed->buf[eol] != '\n') eol++;
                lfm_ui_textedit_move(ed, eol, shift);
                continue;
            }
            if (ctrl && ev->utf8_len == 1) {
                char c = ev->utf8[0];
                if (c == 'a' || c == 'A' || c == 1) {
                    ed->sel_anchor = 0;
                    ed->cursor = ed->len;
                    ed->has_sel = (ed->len > 0);
                } else if ((c == 'c' || c == 'C' || c == 3 || c == 'x' || c == 'X' || c == 24) &&
                           ed->has_sel && ctx->backend.clipboard_set) {
                    size_t a, b;
                    lfm_ui_textedit_sel_range(ed, &a, &b);
                    if (b > a) {
                        ctx->backend.clipboard_set(ctx->backend.userdata, ed->buf + a, b - a);
                        if (c == 'x' || c == 'X' || c == 24) {
                            lfm_ui_textedit_delete_sel(ed);
                            res = LFM_UI_EDIT_CHANGED;
                        }
                    }
                } else if ((c == 'v' || c == 'V' || c == 22) && ctx->backend.clipboard_get) {
                    char clip[8192];
                    size_t clen = ctx->backend.clipboard_get(ctx->backend.userdata, clip, sizeof(clip) - 1);
                    if (clen > 0) {
                        /* Normalize CRLF -> LF */
                        size_t out_n = 0;
                        for (size_t j = 0; j < clen; ++j) {
                            if (clip[j] == '\r') continue;
                            clip[out_n++] = clip[j];
                        }
                        lfm_ui_textedit_insert(ed, clip, out_n);
                        res = LFM_UI_EDIT_CHANGED;
                    }
                }
                continue;
            }
            if (!ctrl && ev->utf8_len > 0 && (unsigned char)ev->utf8[0] >= 32) {
                lfm_ui_textedit_insert(ed, ev->utf8, ev->utf8_len);
                res = LFM_UI_EDIT_CHANGED;
            }
        }
    }

    /* Draw frame */
    lfm_ui_draw_card(ctx, r, 10, ctx->theme.bg_input,
                     ed->focused ? ctx->theme.border_focus : ctx->theme.border);

    lfm_ui_push_clip(ctx, inner);
    int32_t lh = lfm_ui_line_h(ctx, LFM_UI_FONT_UI) + 3;

    if (ed->len == 0 && placeholder && placeholder[0]) {
        lfm_ui_draw_text_cstr(ctx, LFM_UI_FONT_UI, inner.x, inner.y, ctx->theme.text_muted, placeholder);
    }

    /* First locate cursor y to keep cursor in view */
    int32_t cursor_x = inner.x;
    int32_t cursor_y_content = 0;
    {
        size_t pos = 0;
        int32_t row_y = 0;
        bool found = false;
        while (pos < ed->len) {
            LfmUiWrapLine line;
            size_t next = lfm_ui_next_wrap_line(ctx, LFM_UI_FONT_UI, ed->buf, ed->len, pos, inner.w, &line);
            if (!found && ed->cursor >= line.start && ed->cursor <= line.end &&
                !(ed->cursor == line.end && line.end < next && line.end < ed->len && ed->buf[line.end] != '\n')) {
                cursor_x = inner.x + lfm_ui_text_w(ctx, LFM_UI_FONT_UI, ed->buf + line.start, ed->cursor - line.start);
                cursor_y_content = row_y;
                found = true;
            }
            row_y += lh;
            if (next == ed->len && line.end < ed->len && ed->buf[line.end] == '\n') {
                if (!found && ed->cursor == ed->len) {
                    cursor_x = inner.x;
                    cursor_y_content = row_y;
                    found = true;
                }
                row_y += lh;
                break;
            }
            pos = next;
        }
        if (!found) {
            cursor_x = inner.x;
            cursor_y_content = row_y;
        }
        int32_t total_h = row_y > lh ? row_y : lh;
        int32_t max_scroll = total_h - inner.h;
        if (max_scroll < 0) max_scroll = 0;
        if (ed->focused && res != LFM_UI_EDIT_NONE) {
            if (cursor_y_content < ed->scroll_y) ed->scroll_y = cursor_y_content;
            if (cursor_y_content + lh > ed->scroll_y + inner.h) {
                ed->scroll_y = cursor_y_content + lh - inner.h;
            }
        }
        if (ed->scroll_y > max_scroll) ed->scroll_y = max_scroll;
        if (ed->scroll_y < 0) ed->scroll_y = 0;
    }

    /* Second pass: selection highlights + text */
    size_t sel_a = 0, sel_b = 0;
    lfm_ui_textedit_sel_range(ed, &sel_a, &sel_b);
    {
        size_t pos = 0;
        int32_t draw_y = inner.y - ed->scroll_y;
        while (pos < ed->len) {
            LfmUiWrapLine line;
            size_t next = lfm_ui_next_wrap_line(ctx, LFM_UI_FONT_UI, ed->buf, ed->len, pos, inner.w, &line);
            if (sel_b > sel_a && sel_b > line.start && sel_a < line.end) {
                size_t ha = sel_a > line.start ? sel_a : line.start;
                size_t hb = sel_b < line.end ? sel_b : line.end;
                int32_t sx = inner.x + lfm_ui_text_w(ctx, LFM_UI_FONT_UI, ed->buf + line.start, ha - line.start);
                int32_t sw = lfm_ui_text_w(ctx, LFM_UI_FONT_UI, ed->buf + ha, hb - ha);
                if (sw < 4) sw = 4;
                lfm_ui_draw_rect(ctx, lfm_rect(sx, draw_y, sw, lh - 2), ctx->theme.selection);
            }
            if (line.end > line.start) {
                lfm_ui_draw_text(ctx, LFM_UI_FONT_UI, inner.x, draw_y, ctx->theme.text,
                                 ed->buf + line.start, line.end - line.start);
            }
            draw_y += lh;
            pos = next;
        }
    }

    if (ed->focused) {
        int32_t cy = inner.y + cursor_y_content - ed->scroll_y;
        lfm_ui_draw_rect(ctx, lfm_rect(cursor_x, cy + 1, 2, lh - 4), ctx->theme.accent);
    }

    lfm_ui_pop_clip(ctx);
    return res;
}

#endif /* LFM_UI_IMPLEMENTATION */
#endif /* LFM_UI_H */

