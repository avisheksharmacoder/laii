package ai.liquid.lfm2.app

import ai.liquid.lfm2.GenerationChunk

/**
 * Android Activity controller implementing standard mobile lifecycle hooks.
 * <p>
 * In production Android apps with Jetpack, this extends {@code androidx.activity.ComponentActivity}.
 */
class MainActivity(private val viewModel: ChatViewModel = ChatViewModel()) {

    var isPaused: Boolean = false
        private set

    fun onCreate(modelPath: String) {
        viewModel.initialize(modelPath)
    }

    fun onResume(): Double {
        isPaused = false
        return viewModel.onAppForegrounded()
    }

    fun onPause(): Double {
        isPaused = true
        return viewModel.onAppBackgrounded()
    }

    fun onUserPrompt(prompt: String, maxChunks: Int = 2000, onChunk: (GenerationChunk) -> Unit) {
        if (isPaused) {
            onResume()
        }
        viewModel.sendMessage(prompt, maxChunks = maxChunks, onChunk = onChunk)
    }

    fun onDestroy() {
        viewModel.onCleared()
    }

    fun getViewModel(): ChatViewModel = viewModel
}
