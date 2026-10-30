package io.github.chaitanyakash.cyberlens

import androidx.activity.compose.PredictiveBackHandler
import androidx.activity.BackEventCompat
import androidx.compose.animation.core.Animatable
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import io.github.chaitanyakash.cyberlens.icons.*
import androidx.compose.material3.*
import androidx.compose.material3.adaptive.currentWindowAdaptiveInfo
import androidx.compose.material3.adaptive.navigationsuite.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.saveable.rememberSaveableStateHolder
import androidx.compose.ui.*
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.*
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.*
import androidx.compose.ui.tooling.preview.*
import androidx.compose.ui.unit.*
import io.github.chaitanyakash.cyberlens.core.designsystem.*
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.collect

enum class Destination(val label: Int, val icon: ImageVector) {
    Protection(R.string.protection, Icons.Rounded.Shield),
    Messages(R.string.messages, Icons.Rounded.ChatBubbleOutline),
    Check(R.string.check, Icons.Rounded.AddCircleOutline),
    History(R.string.history, Icons.Rounded.History),
    Profile(R.string.profile, Icons.Rounded.PersonOutline),
}

@Composable
fun CyberLensShell(
    dark: Boolean = isSystemInDarkTheme(),
    reduceMotion: Boolean = false,
    reduceTransparency: Boolean = false,
    onMotion: (Boolean) -> Unit = {},
    onTransparency: (Boolean) -> Unit = {},
    storage: StorageState = StorageState.Starting,
    onEmergency: () -> Unit = {},
    initialDestination: Destination = Destination.Protection,
    initialPage: String = "",
    initialCatalogueSection: Int = 0,
    initialCatalogueSheet: Boolean = false,
    showDebug: Boolean = BuildConfig.DEBUG,
) {
    var destination by rememberSaveable { mutableStateOf(initialDestination) }
    var page by rememberSaveable { mutableStateOf(initialPage) }
    val savedScreens = rememberSaveableStateHolder()
    val backProgress = remember { Animatable(0f) }
    val scope = rememberCoroutineScope()
    var backDirection by remember { mutableFloatStateOf(1f) }
    val modalProgress = remember { mutableFloatStateOf(0f) }
    PredictiveBackHandler(page.isNotEmpty() || destination != Destination.Protection) { events ->
        val start = backProgress.value
        try {
            events.collect { event ->
                backDirection = if (event.swipeEdge == BackEventCompat.EDGE_LEFT) 1f else -1f
                backProgress.snapTo(start + (1f - start) * event.progress)
            }
            if (page.isNotEmpty()) page = "" else destination = Destination.Protection
            backProgress.snapTo(0f)
        } catch (_: CancellationException) {
            scope.launch { backProgress.animateTo(0f, CLMotion.default) }
        }
    }
    CyberLensTheme(dark, reduceMotion, reduceTransparency) {
        CompositionLocalProvider(LocalModalProgress provides modalProgress) {
        Surface(Modifier.fillMaxSize().graphicsLayer {
            val progress = backProgress.value
            val modal = modalProgress.floatValue
            if (reduceMotion) alpha = (1f - 0.2f * progress) * (1f - 0.05f * modal)
            else {
                scaleX = (1f - 0.04f * progress) * (1f - 0.04f * modal)
                scaleY = scaleX
                translationX = backDirection * size.width * 0.2f * progress
            }
            shape = RoundedCornerShape(24.dp)
            clip = modal > 0f
        }, color = MaterialTheme.colorScheme.background) {
            BoxWithConstraints {
                val margin = if (maxWidth >= 840.dp) 32.dp else if (maxWidth >= 600.dp) 24.dp else 16.dp
                val select: (Destination) -> Unit = { destination = it; page = "" }
                val content: @Composable () -> Unit = {
                    savedScreens.SaveableStateProvider(page.ifEmpty { destination.name }) {
                    when(page) {
                        "components" -> ComponentCatalogue(onBack = { page = "" }, initialDark = dark,
                            initialSection = initialCatalogueSection, initialSheet = initialCatalogueSheet)
                        "settings" -> SettingsScreen(reduceMotion, reduceTransparency, onMotion, onTransparency) { page = "" }
                        else -> ShellContent(destination, margin, storage, onEmergency,
                            onSettings = { page = "settings" }, onComponents = { page = "components" }, showDebug = showDebug)
                    }
                    }
                }
                AdaptiveNavigation(destination, select) { content() }

            }
        }
        }
    }
}

@Composable
private fun ShellContent(destination: Destination, margin: Dp, storage: StorageState,
    onEmergency: () -> Unit, onSettings: () -> Unit, onComponents: () -> Unit, showDebug: Boolean) {
    val wordmark: (@Composable () -> Unit)? = if (destination == Destination.Protection) {
        {
            Column {
                CLWordmark()
                Text(stringResource(R.string.shell_preview), style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    } else null
    CLLargeTitleScaffold(
        title = stringResource(if (destination == Destination.Protection) R.string.app_name else destination.label),
        expandedTitle = wordmark,
        maxContentWidth = if (destination == Destination.Protection) Dp.Unspecified else 640.dp,
        sideMargin = margin,
    ) {
        if (destination != Destination.Protection)
            Text(stringResource(R.string.shell_preview), style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        when(destination) {
            Destination.Protection -> {
                BoxWithConstraints(Modifier.fillMaxWidth()) {
                val status: @Composable () -> Unit = { CLCard(Modifier.fillMaxWidth()) {
                    val protectionProgress = stringResource(R.string.protection_progress, 0, 4)
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                        Box(Modifier.size(64.dp).semantics { contentDescription = protectionProgress }, contentAlignment = Alignment.Center) {
                            CircularProgressIndicator(progress = { 0f }, modifier = Modifier.fillMaxSize(),
                                strokeWidth = 5.dp, trackColor = MaterialTheme.colorScheme.surfaceVariant)
                            Icon(Icons.Rounded.Shield, null, Modifier.size(28.dp), tint = MaterialTheme.colorScheme.primary)
                        }
                        Text(stringResource(R.string.setup_title), Modifier.weight(1f),
                            style = MaterialTheme.typography.headlineMedium)
                    }
                    Text(stringResource(R.string.setup_body), style = MaterialTheme.typography.bodyLarge)
                    Text(stringResource(when(storage) {
                        StorageState.Starting -> R.string.database_starting
                        StorageState.Ready -> R.string.database_ready
                        StorageState.Error -> R.string.database_error
                    }), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                } }
                val capabilities: @Composable () -> Unit = { CLGroupedList {
                    CLListRow(stringResource(R.string.recording_status), stringResource(R.string.recording_detail))
                    CLListRow(stringResource(R.string.messages_status), stringResource(R.string.messages_detail))
                    CLListRow(stringResource(R.string.models_status), stringResource(R.string.models_detail))
                } }
                if (maxWidth >= 664.dp) {
                    Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
                        Box(Modifier.weight(1f)) { status() }
                        Box(Modifier.weight(1f)) { capabilities() }
                    }
                } else {
                    Column(verticalArrangement = Arrangement.spacedBy(24.dp)) {
                        status()
                        capabilities()
                    }
                }
                }
                Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                    Box(Modifier.widthIn(max = 640.dp)) {
                        CLEmergencyStrip(stringResource(R.string.emergency), stringResource(R.string.call_1930), onEmergency)
                    }
                }
            }
            Destination.Messages -> CLEmptyState(stringResource(R.string.empty_messages), stringResource(R.string.empty_messages_body))
            Destination.History -> CLEmptyState(stringResource(R.string.empty_history), stringResource(R.string.empty_history_body))
            Destination.Check -> {
                Text(stringResource(R.string.check_title), style = MaterialTheme.typography.headlineMedium)
                CLGroupedList {
                    listOf(R.string.paste, R.string.screenshot, R.string.audio, R.string.describe).forEach {
                        CLListRow(stringResource(it), stringResource(R.string.later))
                    }
                }
                Text(stringResource(R.string.share_hint), style = MaterialTheme.typography.bodyLarge)
            }
            Destination.Profile -> {
                CLEmptyState(stringResource(R.string.profile_title), stringResource(R.string.profile_body))
                CLPrimaryButton(stringResource(R.string.settings), onSettings)
                Text(stringResource(R.string.publisher), style = MaterialTheme.typography.bodySmall)
            }
        }
        if (showDebug) CLSecondaryButton(stringResource(R.string.components), onComponents)
        Spacer(Modifier.height(16.dp))
    }
}

@Composable
private fun SettingsScreen(motion: Boolean, transparency: Boolean, onMotion: (Boolean) -> Unit,
    onTransparency: (Boolean) -> Unit, onBack: () -> Unit) {
    CLLargeTitleScaffold(stringResource(R.string.settings),
        navigation = { CLSecondaryButton(stringResource(R.string.back), onBack) }) {
        Text(stringResource(R.string.appearance), style = MaterialTheme.typography.headlineMedium)
        CLGroupedList {
            CLToggleRow(stringResource(R.string.reduce_motion), motion, onMotion,
                stringResource(R.string.reduce_motion_detail))
            CLToggleRow(stringResource(R.string.reduce_transparency), transparency, onTransparency,
                stringResource(R.string.reduce_transparency_detail))
        }
    }
}

@PreviewScreenSizes
@PreviewFontScale
@Composable
private fun ShellPreview() { CyberLensShell(showDebug = true) }

@PreviewScreenSizes
@PreviewFontScale
@Composable
private fun MessagesPreview() { CyberLensShell(initialDestination = Destination.Messages) }

@PreviewScreenSizes
@PreviewFontScale
@Composable
private fun CheckPreview() { CyberLensShell(initialDestination = Destination.Check) }

@PreviewScreenSizes
@PreviewFontScale
@Composable
private fun HistoryPreview() { CyberLensShell(initialDestination = Destination.History) }

@PreviewScreenSizes
@PreviewFontScale
@Composable
private fun ProfilePreview() { CyberLensShell(initialDestination = Destination.Profile) }

@PreviewScreenSizes
@PreviewFontScale
@Composable
private fun SettingsPreview() { CyberLensShell(initialPage = "settings") }
