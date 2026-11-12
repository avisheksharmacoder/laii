/*
 * test_lfm2_gui.c — Headless unit & integration tests for the native desktop GUI stack
 * (include/lfm_ui.h + tools/lfm2_gui_worker.h + liblfm2.so).
 *
 * Runs cleanly without requiring an X11 display server.
 */

#define LFM_UI_IMPLEMENTATION
#include "lfm_ui.h"
#include "lfm2_gui_worker.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    int32_t rects_drawn;
    int32_t cards_drawn;
    int32_t texts_drawn;
    char    clipboard[1024];
    size_t  clipboard_len;
} MockUiState;

static void mock_draw_rect(void *ud, LfmUiRect r, LfmUiColor c) {
    (void)r; (void)c;
    ((MockUiState *)ud)->rects_drawn++;
}

static void mock_draw_rounded_rect(void *ud, LfmUiRect r, int32_t rad,
                                   LfmUiColor fill, LfmUiColor border, int32_t bw) {
    (void)r; (void)rad; (void)fill; (void)border; (void)bw;
    ((MockUiState *)ud)->cards_drawn++;
}

static void mock_set_clip(void *ud, LfmUiRect r) {
    (void)ud; (void)r;
}

static int32_t mock_measure_text(void *ud, LfmUiFontRole font, const char *s, size_t len) {
    (void)ud; (void)font;
    if (!s || len == 0) return 0;
    size_t glyphs = 0;
    for (size_t i = 0; i < len; ) {
        i = lfm_utf8_next(s, len, i);
        glyphs++;
    }
    return (int32_t)(glyphs * 8);
}

static int32_t mock_line_height(void *ud, LfmUiFontRole font) {
    (void)ud; (void)font;
    return 16;
}

static void mock_draw_text(void *ud, LfmUiFontRole font, int32_t x, int32_t y,
                           LfmUiColor c, const char *s, size_t len) {
    (void)font; (void)x; (void)y; (void)c; (void)s; (void)len;
    ((MockUiState *)ud)->texts_drawn++;
}

static void mock_clip_set(void *ud, const char *s, size_t len) {
    MockUiState *m = (MockUiState *)ud;
    size_t copy = len < sizeof(m->clipboard) - 1 ? len : sizeof(m->clipboard) - 1;
    memcpy(m->clipboard, s, copy);
    m->clipboard[copy] = '\0';
    m->clipboard_len = copy;
}

static size_t mock_clip_get(void *ud, char *out, size_t cap) {
    MockUiState *m = (MockUiState *)ud;
    size_t copy = m->clipboard_len < cap ? m->clipboard_len : (cap - 1);
    memcpy(out, m->clipboard, copy);
    out[copy] = '\0';
    return copy;
}

static uint64_t mock_now_ms(void *ud) {
    (void)ud;
    return 1000;
}

static LfmUiBackend make_mock_backend(MockUiState *st) {
    memset(st, 0, sizeof(*st));
    LfmUiBackend be;
    memset(&be, 0, sizeof(be));
    be.userdata = st;
    be.draw_rect = mock_draw_rect;
    be.draw_rounded_rect = mock_draw_rounded_rect;
    be.set_clip = mock_set_clip;
    be.measure_text = mock_measure_text;
    be.line_height = mock_line_height;
    be.draw_text = mock_draw_text;
    be.clipboard_set = mock_clip_set;
    be.clipboard_get = mock_clip_get;
    be.now_ms = mock_now_ms;
    return be;
}

static void test_layout_and_markdown(void) {
    MockUiState st;
    LfmUiContext ctx;
    lfm_ui_init(&ctx, make_mock_backend(&st), true);
    lfm_ui_begin_frame(&ctx, 800, 600);

    LfmUiRect root = lfm_rect(0, 0, 800, 600);
    LfmUiRect top = lfm_rect_cut_top(&root, 50);
    LfmUiRect bot = lfm_rect_cut_bottom(&root, 100);
    LfmUiRect right = lfm_rect_cut_right(&root, 200);
    assert(top.h == 50 && bot.y == 500 && bot.h == 100);
    assert(right.x == 600 && right.w == 200 && root.w == 600 && root.h == 450);

    const char *wrap_txt = "hello world from native desktop gui";
    int32_t max_lw = 0;
    int32_t h = lfm_ui_measure_wrapped_text(&ctx, LFM_UI_FONT_UI, wrap_txt, strlen(wrap_txt), 100, &max_lw);
    assert(h >= 38 && max_lw <= 100);

    const char *md = "Here is C code:\n```c\nint x = 1;\nreturn x;\n```\nDone.";
    int32_t mh = lfm_ui_draw_markdown(&ctx, 10, 10, 320, md, strlen(md), false);
    assert(mh > 60 && st.cards_drawn >= 1 && st.texts_drawn >= 4);
    lfm_ui_end_frame(&ctx);
    printf("[PASS] Layout geometry, UTF-8 wrapping, and markdown code block rendering\n");
}

static void test_textedit_utf8_and_clipboard(void) {
    MockUiState st;
    LfmUiContext ctx;
    lfm_ui_init(&ctx, make_mock_backend(&st), true);

    LfmUiTextEdit ed;
    lfm_ui_textedit_init(&ed, 64);
    uint32_t id = lfm_ui_hash_str("edit");
    ctx.focus_id = id;
    LfmUiRect box = lfm_rect(10, 10, 300, 80);

    lfm_ui_begin_frame(&ctx, 400, 200);
    lfm_ui_push_key(&ctx, LFM_UI_KEY_NONE, 0, "hi ", 3);
    lfm_ui_push_key(&ctx, LFM_UI_KEY_NONE, 0, "\xF0\x9F\x9A\x80", 4); /* rocket UTF-8 */
    lfm_ui_push_key(&ctx, LFM_UI_KEY_ENTER, LFM_UI_MOD_SHIFT, "\n", 1);
    lfm_ui_push_key(&ctx, LFM_UI_KEY_NONE, 0, "ok!", 3);
    lfm_ui_push_key(&ctx, LFM_UI_KEY_BACKSPACE, 0, "", 0);
    assert(lfm_ui_textedit_widget(&ctx, id, &ed, box, "Type...", true) == LFM_UI_EDIT_CHANGED);
    lfm_ui_end_frame(&ctx);
    assert(strcmp(ed.buf, "hi \xF0\x9F\x9A\x80\nok") == 0);

    /* Select all, copy, clear, paste, then Enter submit */
    lfm_ui_begin_frame(&ctx, 400, 200);
    lfm_ui_push_key(&ctx, LFM_UI_KEY_NONE, LFM_UI_MOD_CTRL, "a", 1);
    lfm_ui_push_key(&ctx, LFM_UI_KEY_NONE, LFM_UI_MOD_CTRL, "c", 1);
    lfm_ui_push_key(&ctx, LFM_UI_KEY_BACKSPACE, 0, "", 0);
    lfm_ui_textedit_widget(&ctx, id, &ed, box, "Type...", true);
    lfm_ui_end_frame(&ctx);
    assert(ed.len == 0 && strcmp(st.clipboard, "hi \xF0\x9F\x9A\x80\nok") == 0);

    lfm_ui_begin_frame(&ctx, 400, 200);
    lfm_ui_push_key(&ctx, LFM_UI_KEY_NONE, LFM_UI_MOD_CTRL, "v", 1);
    lfm_ui_push_key(&ctx, LFM_UI_KEY_ENTER, 0, "\n", 1);
    assert(lfm_ui_textedit_widget(&ctx, id, &ed, box, "Type...", true) == LFM_UI_EDIT_SUBMIT);
    lfm_ui_end_frame(&ctx);
    assert(strcmp(ed.buf, "hi \xF0\x9F\x9A\x80\nok") == 0);

    lfm_ui_textedit_free(&ed);
    printf("[PASS] Multi-line UTF-8 TextEdit, selection, clipboard, and Enter submit\n");
}

static void test_sse_think_and_dir_scan(void) {
    char delta[256], fin[32];
    size_t dlen = 0;
    bool done = false;

    const char *l1 = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\\n\\\"LFM\\\"\"}}]}";
    assert(lfm_gui_extract_sse_line(l1, delta, sizeof(delta), &dlen, fin, sizeof(fin), &done));
    assert(!done && strcmp(delta, "Hello\n\"LFM\"") == 0);

    const char *l2 = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}";
    assert(lfm_gui_extract_sse_line(l2, delta, sizeof(delta), &dlen, fin, sizeof(fin), &done));
    assert(!done && strcmp(fin, "length") == 0);

    const char *l3 = "data: [DONE]";
    assert(lfm_gui_extract_sse_line(l3, delta, sizeof(delta), &dlen, fin, sizeof(fin), &done));
    assert(done);

    LfmGuiThinkParser tp;
    lfm_gui_think_parser_init(&tp);
    LfmGuiString th, ans;
    lfm_gui_str_init(&th);
    lfm_gui_str_init(&ans);
    lfm_gui_think_parser_feed(&tp, "<thi", 4, &th, &ans);
    lfm_gui_think_parser_feed(&tp, "nk>plan step</th", 16, &th, &ans);
    lfm_gui_think_parser_feed(&tp, "ink>Final answer!", 17, &th, &ans);
    lfm_gui_think_parser_flush(&tp, &th, &ans);
    assert(th.data && strcmp(th.data, "plan step") == 0);
    assert(ans.data && strcmp(ans.data, "Final answer!") == 0);
    lfm_gui_str_free(&th);
    lfm_gui_str_free(&ans);

    LfmGuiFileEntry entries[64];
    size_t n = lfm_gui_scan_dir("tests", entries, 64);
    bool found_toy = false;
    for (size_t i = 0; i < n; ++i) {
        if (strcmp(entries[i].name, "lfm2_toy.gguf") == 0 && !entries[i].is_dir) {
            found_toy = true;
        }
    }
    assert(found_toy);
    printf("[PASS] SSE JSON extraction, split-chunk <think> routing, and GGUF directory scanner\n");
}

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    int             wakes;
} WakeLatch;

static void latch_wake(void *ud) {
    WakeLatch *w = (WakeLatch *)ud;
    pthread_mutex_lock(&w->mu);
    w->wakes++;
    pthread_cond_broadcast(&w->cv);
    pthread_mutex_unlock(&w->mu);
}

static void test_worker_local_inference(void) {
    LfmGuiConfig cfg;
    lfm_gui_config_defaults(&cfg);
    assert(cfg.ctx_len == 10000);
    assert(cfg.max_tokens == 2000);

    snprintf(cfg.model_path, sizeof(cfg.model_path), "tests/lfm2_toy.gguf");
    cfg.max_tokens = 12;
    cfg.temperature = 0.0f;

    WakeLatch latch;
    pthread_mutex_init(&latch.mu, NULL);
    pthread_cond_init(&latch.cv, NULL);
    latch.wakes = 0;

    LfmGuiEngine eng;
    lfm_gui_engine_init(&eng, &cfg, latch_wake, &latch);
    assert(lfm_gui_engine_send_prompt(&eng, "Hello LFM"));

    for (int iter = 0; iter < 100; ++iter) {
        pthread_mutex_lock(&eng.mu);
        bool done = (eng.n_msgs == 2 && !eng.msgs[1].streaming && eng.status != LFM_GUI_GENERATING &&
                     eng.status != LFM_GUI_LOADING_MODEL);
        pthread_mutex_unlock(&eng.mu);
        if (done) break;
        usleep(20000);
    }

    pthread_mutex_lock(&eng.mu);
    assert(eng.model_loaded);
    assert(eng.n_msgs == 2);
    assert(!eng.msgs[1].streaming);
    assert(eng.msgs[1].gen_tokens > 0 && eng.msgs[1].gen_tokens <= 12);
    assert(strcmp(eng.msgs[1].stop_reason, "stop") == 0 ||
           strcmp(eng.msgs[1].stop_reason, "length") == 0);
    pthread_mutex_unlock(&eng.mu);

    lfm_gui_engine_reset_chat(&eng);
    pthread_mutex_lock(&eng.mu);
    assert(eng.n_msgs == 0);
    pthread_mutex_unlock(&eng.mu);

    lfm_gui_engine_destroy(&eng);
    pthread_cond_destroy(&latch.cv);
    pthread_mutex_destroy(&latch.mu);
    printf("[PASS] GUI worker pthread + liblfm2.so local generation & reset\n");
}

int main(void) {
    test_layout_and_markdown();
    test_textedit_utf8_and_clipboard();
    test_sse_think_and_dir_scan();
    test_worker_local_inference();
    printf("ALL NATIVE GUI TESTS PASSED!\n");
    return 0;
}

