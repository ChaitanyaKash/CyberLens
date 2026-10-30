package io.github.chaitanyakash.cyberlens.core.testing

data class TestWindow(val name: String, val width: Int, val height: Int)
object ScreenMatrix {
    val windows = listOf(
        TestWindow("small_phone", 320, 640),
        TestWindow("phone", 384, 832),
        TestWindow("phone_landscape", 832, 384),
        TestWindow("wide_phone", 411, 891),
        TestWindow("large_phone", 432, 960),
        TestWindow("large_phone_landscape", 960, 432),
        TestWindow("tablet_portrait", 800, 1280),
        TestWindow("tablet_landscape", 1280, 800),
        TestWindow("split_screen", 360, 800),
    )
}
