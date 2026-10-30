@file:OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)

package io.github.chaitanyakash.cyberlens.core.designsystem

import android.os.Build
import android.view.WindowManager
import androidx.activity.compose.LocalOnBackPressedDispatcherOwner
import androidx.activity.compose.PredictiveBackHandler
import androidx.activity.findViewTreeOnBackPressedDispatcherOwner
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.gestures.LocalBringIntoViewSpec
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.*
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.util.VelocityTracker
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.*
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.*
import androidx.compose.ui.unit.*
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.compose.ui.window.DialogWindowProvider
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import java.util.function.Consumer
import kotlin.math.roundToInt

@Composable
fun CLQuestionSheet(title: String, onDismiss: () -> Unit,
    content: @Composable ColumnScope.(dismiss: () -> Unit) -> Unit) {
    val callerDensity = LocalDensity.current
    val scope = rememberCoroutineScope()
    val motion = LocalReduceMotion.current
    val position = remember { Animatable(1000f) }
    var height by remember { mutableFloatStateOf(1000f) }
    var dragging by remember { mutableStateOf(false) }
    var dragPosition by remember { mutableFloatStateOf(0f) }
    var rawDragPosition by remember { mutableFloatStateOf(0f) }
    var backHandling by remember { mutableStateOf(false) }
    var animation by remember { mutableStateOf<Job?>(null) }
    val velocity = remember { VelocityTracker() }
    val haptic = LocalHapticFeedback.current
    val modalProgress = LocalModalProgress.current
    val reduceTransparency = LocalReduceTransparency.current
    val handleLabel = stringResource(R.string.cl_sheet_handle)
    val dismissLabel = stringResource(R.string.cl_dismiss_sheet)
    val fullWindowFocusSpec = LocalUnobscuredBringIntoViewSpec.current ?: LocalBringIntoViewSpec.current
    fun visibleProgress() = (1f - (if (dragging) dragPosition else position.value) / height).coerceIn(0f, 1f)
    fun dismiss() {
        val fromDrag = dragging
        val start = dragPosition
        val inheritedVelocity = if (motion) 0f else if (fromDrag) velocity.calculateVelocity().y else position.velocity
        dragging = false
        animation?.cancel()
        animation = scope.launch {
            if (fromDrag) position.snapTo(start)
            position.animateTo(height, if (motion) tween(180) else CLMotion.sheet, inheritedVelocity)
            onDismiss()
        }
    }
    CompositionLocalProvider(LocalBringIntoViewSpec provides fullWindowFocusSpec) {
    Dialog(onDismissRequest = { dismiss() }, properties = DialogProperties(
        usePlatformDefaultWidth = false, decorFitsSystemWindows = false,
    )) {
        CompositionLocalProvider(LocalDensity provides callerDensity) {
        val view = LocalView.current
        val window = (view.parent as? DialogWindowProvider)?.window
        val canBlur = windowBlurSupported() && !reduceTransparency
        val radius = with(LocalDensity.current) { 24.dp.roundToPx() }
        DisposableEffect(Unit) { onDispose { modalProgress?.value = 0f } }
        DisposableEffect(window, canBlur) {
            var originalRadius = 0
            var hadBlurFlag = false
            val hadDimFlag = window != null && window.attributes.flags and WindowManager.LayoutParams.FLAG_DIM_BEHIND != 0
            window?.clearFlags(WindowManager.LayoutParams.FLAG_DIM_BEHIND)
            if (Build.VERSION.SDK_INT >= 31 && canBlur && window != null) {
                originalRadius = window.attributes.blurBehindRadius
                hadBlurFlag = window.attributes.flags and WindowManager.LayoutParams.FLAG_BLUR_BEHIND != 0
                window.addFlags(WindowManager.LayoutParams.FLAG_BLUR_BEHIND)
            }
            onDispose {
                if (hadDimFlag) window?.addFlags(WindowManager.LayoutParams.FLAG_DIM_BEHIND)
                if (Build.VERSION.SDK_INT >= 31 && canBlur && window != null) {
                    window.attributes = window.attributes.apply { blurBehindRadius = originalRadius }
                    if (!hadBlurFlag) window.clearFlags(WindowManager.LayoutParams.FLAG_BLUR_BEHIND)
                }
            }
        }
        LaunchedEffect(window, canBlur) {
            snapshotFlow { visibleProgress() }.collect { progress ->
                modalProgress?.value = progress
                if (Build.VERSION.SDK_INT >= 31 && canBlur && window != null)
                    window.attributes = window.attributes.apply { blurBehindRadius = (radius * progress).roundToInt() }
            }
        }
        val backOwner = view.findViewTreeOnBackPressedDispatcherOwner()
        if (backOwner != null) CompositionLocalProvider(LocalOnBackPressedDispatcherOwner provides backOwner) {
            PredictiveBackHandler { events ->
                backHandling = true
                animation?.cancel()
                if (dragging) position.snapTo(dragPosition)
                dragging = false
                val start = position.value
                try {
                    events.collect { position.snapTo(start + (height - start) * it.progress) }
                    dismiss()
                } catch (_: CancellationException) {
                    animation = scope.launch { position.animateTo(start, if (motion) tween(180) else CLMotion.sheet) }
                } finally {
                    backHandling = false
                }
            }
        }
        Box(Modifier.fillMaxSize().imePadding().drawBehind {
            drawRect(androidx.compose.ui.graphics.Color.Black.copy(alpha = 0.32f * visibleProgress()))
        }) {
            Box(Modifier.fillMaxSize().clickable(indication = null,
                interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() }) { dismiss() }
                .clearAndSetSemantics { })
            CLModalPane {
            Column(Modifier.align(Alignment.BottomCenter).widthIn(max = 640.dp).fillMaxWidth()
                .onSizeChanged { height = it.height.toFloat() }
                .graphicsLayer {
                    val visiblePosition = if (dragging) dragPosition else position.value
                    translationY = if (motion) 0f else visiblePosition
                    if (motion) alpha = (1f - visiblePosition / height).coerceIn(0f, 1f)
                }
                .background(MaterialTheme.colorScheme.surface.copy(alpha =
                    if (reduceTransparency) 1f else if (canBlur) 0.85f else 0.95f),
                    RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp))
                .then(if (reduceTransparency) Modifier.border(1.dp, MaterialTheme.colorScheme.outline,
                    RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)) else Modifier)
                .semantics { paneTitle = title }
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .verticalScroll(rememberScrollState())
                .padding(24.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Box(Modifier.fillMaxWidth().height(48.dp)
                    .semantics {
                        contentDescription = handleLabel
                        customActions = listOf(CustomAccessibilityAction(dismissLabel) { dismiss(); true })
                    }
                    .pointerInput(motion) {
                        detectVerticalDragGestures(
                            onDragStart = {
                                if (!backHandling) {
                                animation?.cancel()
                                dragPosition = position.value
                                rawDragPosition = dragPosition
                                dragging = true
                                velocity.resetTracking()
                                }
                            },
                            onVerticalDrag = { change, delta ->
                                if (dragging) {
                                change.consume()
                                velocity.addPosition(change.uptimeMillis, change.position + Offset(0f, dragPosition))
                                rawDragPosition += delta
                                dragPosition = when {
                                    rawDragPosition < 0f -> CLMotion.rubberband(rawDragPosition, height)
                                    rawDragPosition > height -> height + CLMotion.rubberband(rawDragPosition - height, height)
                                    else -> rawDragPosition
                                }
                                }
                            },
                            onDragEnd = {
                                if (dragging) {
                                val released = velocity.calculateVelocity().y
                                val target = if (dragPosition + CLMotion.project(released) > height * 0.5f) height else 0f
                                val start = dragPosition
                                animation = scope.launch {
                                    position.snapTo(start)
                                    dragging = false
                                    haptic.performHapticFeedback(androidx.compose.ui.hapticfeedback.HapticFeedbackType.SegmentTick)
                                    position.animateTo(target, if (motion) tween(180) else if (released > 100f) CLMotion.flickedSheet else CLMotion.sheet, released)
                                    if (target == height) onDismiss()
                                }
                                }
                            },
                            onDragCancel = {
                                if (dragging) {
                                val start = dragPosition
                                animation = scope.launch {
                                    position.snapTo(start)
                                    dragging = false
                                    position.animateTo(0f, if (motion) tween(180) else CLMotion.sheet)
                                }
                                }
                            },
                        )
                    }, contentAlignment = Alignment.Center) {
                    Box(Modifier.size(36.dp, 5.dp).background(MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f), RoundedCornerShape(50)))
                }
                Text(title, Modifier.semantics { heading() }, style = MaterialTheme.typography.headlineMedium)
                content { dismiss() }
            }
            }
        }
        LaunchedEffect(Unit) {
            position.snapTo(height)
            animation = scope.launch {
                position.animateTo(0f, if (motion) tween(180) else CLMotion.sheet)
            }
        }
    }
    }
    }
}

@Composable
private fun windowBlurSupported(): Boolean {
    val context = LocalContext.current
    var supported by remember { mutableStateOf(false) }
    DisposableEffect(context) {
        if (Build.VERSION.SDK_INT >= 31) {
            val manager = context.getSystemService(WindowManager::class.java)
            val listener = Consumer<Boolean> { supported = it }
            var registered = false
            try {
                manager?.addCrossWindowBlurEnabledListener(listener)
                registered = manager != null
                supported = manager?.isCrossWindowBlurEnabled == true
            } catch (_: RuntimeException) { supported = false }
            onDispose {
                if (registered) try { manager?.removeCrossWindowBlurEnabledListener(listener) }
                catch (_: RuntimeException) { }
            }
        } else onDispose { }
    }
    return supported
}
