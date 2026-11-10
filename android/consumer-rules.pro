# Consumer ProGuard / R8 rules for LFM2 Native Engine (ai.liquid.lfm2)
# JNI_OnLoad in liblfm2.so resolves LFM2Model and LFM2Context dynamically via FindClass().

-keep class ai.liquid.lfm2.LFM2Model {
    native <methods>;
}

-keep class ai.liquid.lfm2.LFM2Context {
    native <methods>;
}

-keep class ai.liquid.lfm2.** { *; }

