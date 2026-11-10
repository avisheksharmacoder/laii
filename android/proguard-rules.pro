# Module ProGuard / R8 rules for LFM2 library builds
-keep class ai.liquid.lfm2.LFM2Model {
    native <methods>;
}

-keep class ai.liquid.lfm2.LFM2Context {
    native <methods>;
}

-keep class ai.liquid.lfm2.** { *; }

