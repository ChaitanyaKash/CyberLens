package io.github.chaitanyakash.cyberlens

import android.content.Intent
import android.database.ContentObserver
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.ViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.AndroidEntryPoint
import dagger.hilt.android.lifecycle.HiltViewModel
import io.github.chaitanyakash.cyberlens.core.data.*
import javax.inject.Inject
import javax.inject.Provider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

@HiltViewModel
class ShellViewModel @Inject constructor(
    private val appearance: AppearanceSettings,
    private val database: Provider<CyberLensDatabase>,
) : ViewModel() {
    val reduceMotion = appearance.motion.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), false)
    val reduceTransparency = appearance.transparency.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), false)
    private val _storage = MutableStateFlow(StorageState.Starting)
    val storage = _storage.asStateFlow()
    init {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                database.get().incidents().observe().first()
                _storage.value = StorageState.Ready
            } catch (_: Exception) { _storage.value = StorageState.Error }
        }
    }
    fun motion(value: Boolean) { viewModelScope.launch { appearance.setMotion(value) } }
    fun transparency(value: Boolean) { viewModelScope.launch { appearance.setTransparency(value) } }
}
enum class StorageState { Starting, Ready, Error }

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val model: ShellViewModel = viewModel()
            val motion by model.reduceMotion.collectAsStateWithLifecycle()
            val transparency by model.reduceTransparency.collectAsStateWithLifecycle()
            var systemMotion by remember { mutableStateOf(false) }
            var highContrast by remember { mutableStateOf(false) }
            DisposableEffect(Unit) {
                fun update() {
                    systemMotion = Settings.Global.getFloat(contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
                    highContrast = Settings.Secure.getInt(contentResolver, "high_text_contrast_enabled", 0) == 1
                }
                val observer = object : ContentObserver(Handler(Looper.getMainLooper())) {
                    override fun onChange(selfChange: Boolean) = update()
                }
                contentResolver.registerContentObserver(Settings.Global.getUriFor(Settings.Global.ANIMATOR_DURATION_SCALE), false, observer)
                contentResolver.registerContentObserver(Settings.Secure.getUriFor("high_text_contrast_enabled"), false, observer)
                update()
                onDispose { contentResolver.unregisterContentObserver(observer) }
            }
            val storage by model.storage.collectAsStateWithLifecycle()
            CyberLensShell(
                reduceMotion = motion || systemMotion,
                reduceTransparency = transparency || highContrast,
                onMotion = model::motion, onTransparency = model::transparency,
                storage = storage,
                onEmergency = { startActivity(Intent(Intent.ACTION_DIAL, Uri.parse("tel:1930"))) },
            )
        }
    }
}
