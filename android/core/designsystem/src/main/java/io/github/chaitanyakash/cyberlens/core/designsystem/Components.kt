@file:OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)

package io.github.chaitanyakash.cyberlens.core.designsystem

import android.os.Build
import androidx.compose.animation.core.*
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.*
import androidx.compose.foundation.interaction.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.*
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.gestures.BringIntoViewSpec
import androidx.compose.foundation.gestures.LocalBringIntoViewSpec
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.*
import androidx.compose.ui.draw.*
import androidx.compose.ui.graphics.*
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.*
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.*
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.*
import dev.chrisbanes.haze.*

fun Modifier.clPress(
    interactions: MutableInteractionSource,
    reduceMotion: Boolean,
    focusColor: Color? = null,
): Modifier = composed {
    val pressed by interactions.collectIsPressedAsState()
    val hovered by interactions.collectIsHoveredAsState()
    val focused by interactions.collectIsFocusedAsState()
    val ringColor = focusColor ?: MaterialTheme.colorScheme.primary
    val scale by animateFloatAsState(
        if (pressed && !reduceMotion) 0.97f else 1f, CLMotion.micro, label = "press",
    )
    graphicsLayer { scaleX = scale; scaleY = scale; alpha = if (pressed) 0.85f else if (hovered) 0.94f else 1f }
        .drawWithContent {
            drawContent()
            if (focused) drawRoundRect(ringColor, cornerRadius = CornerRadius(24.dp.toPx()), style = Stroke(2.dp.toPx()))
        }
}

@Composable
fun CLPrimaryButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    val interactions = remember { MutableInteractionSource() }
    val colors = MaterialTheme.colorScheme
    Box(modifier.fillMaxWidth().heightIn(min = 52.dp)
        .clPress(interactions, LocalReduceMotion.current, focusColor = colors.onPrimary)
        .clip(RoundedCornerShape(50))
        .background(colors.primary)
        .clickable(interactionSource = interactions, indication = null, role = Role.Button, onClick = onClick)
        .padding(horizontal = 24.dp, vertical = 14.dp), contentAlignment = Alignment.Center) {
        Text(text, style = MaterialTheme.typography.labelLarge, color = colors.onPrimary)
    }
}
@Composable
fun CLSecondaryButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    val source = remember { MutableInteractionSource() }
    Box(modifier.heightIn(min = 48.dp).clPress(source, LocalReduceMotion.current)
        .clip(RoundedCornerShape(50)).clickable(source, indication = null, role = Role.Button, onClick = onClick)
        .padding(horizontal = 16.dp, vertical = 12.dp), contentAlignment = Alignment.Center) {
        Text(text, color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.titleMedium)
    }
}
@Composable
fun CLCard(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    Surface(modifier, shape = RoundedCornerShape(CLTokens.cardRadius),
        color = MaterialTheme.colorScheme.surface,
        border = if (LocalReduceTransparency.current) BorderStroke(1.dp, MaterialTheme.colorScheme.outline) else null) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp), content = content)
    }
}
@Composable
fun CLWordmark(modifier: Modifier = Modifier) {
    val measurer = rememberTextMeasurer()
    val density = LocalDensity.current
    BoxWithConstraints(modifier) {
    val ideal = measurer.measure("CyberLens", TextStyle(fontFamily = CLWordmarkFont,
        fontSize = 28.sp, fontWeight = FontWeight.SemiBold, letterSpacing = (-0.01).em)).size.width
    val available = with(density) { maxWidth.toPx() }
    val fontSize = 28.sp * (available / (ideal + 2f)).coerceIn(0.1f, 1f)
    val wordmarkStyle = TextStyle(fontFamily = CLWordmarkFont, fontWeight = FontWeight.SemiBold,
        fontSize = fontSize, lineHeight = fontSize * 1.2f, letterSpacing = (-0.01).em)
    Row(Modifier.semantics(mergeDescendants = true) {}) {
        Text("Cyber", style = wordmarkStyle, maxLines = 1, softWrap = false)
        Text("Lens", style = wordmarkStyle.copy(
            brush = Brush.linearGradient(listOf(CLTokens.blue, CLTokens.indigo))), maxLines = 1, softWrap = false)
    }
    }
}
@Composable
fun CLChip(text: String, modifier: Modifier = Modifier) {
    Surface(modifier, color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(CLTokens.chipRadius)) {
        Text(text, Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            style = MaterialTheme.typography.labelMedium)
    }
}
@Composable
fun CLVerdictCard(title: String, detail: String, modifier: Modifier = Modifier) {
    CLCard(modifier.border(1.dp, CLTokens.amber.copy(alpha = 0.4f), RoundedCornerShape(16.dp))) {
        Text(title, style = MaterialTheme.typography.headlineMedium)
        Text(detail, style = MaterialTheme.typography.bodyLarge)
    }
}
@Composable
fun CLListRow(title: String, detail: String, modifier: Modifier = Modifier, trailing: @Composable () -> Unit = {}) {
    Row(modifier.fillMaxWidth().heightIn(min = 56.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, style = MaterialTheme.typography.titleLarge)
            if (detail.isNotBlank()) Text(detail, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        trailing()
    }
}
@Composable
fun CLGroupedList(content: @Composable ColumnScope.() -> Unit) = CLCard(content = content)

@Composable
fun CLToggleRow(title: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit, detail: String = "", enabled: Boolean = true) {
    val source = remember { MutableInteractionSource() }
    val haptics = LocalHapticFeedback.current
    Row(Modifier.fillMaxWidth().heightIn(min = 56.dp)
        .clPress(source, LocalReduceMotion.current)
        .toggleable(value = checked, enabled = enabled, interactionSource = source, indication = null, role = Role.Switch,
            onValueChange = {
                onCheckedChange(it)
                haptics.performHapticFeedback(if (it) HapticFeedbackType.ToggleOn else HapticFeedbackType.ToggleOff)
            }).padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, style = MaterialTheme.typography.titleLarge)
            if (detail.isNotBlank()) Text(detail, style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        CLSwitch(checked, enabled)
    }
}

@Composable
private fun CLSwitch(checked: Boolean, enabled: Boolean) {
    val reduced = LocalReduceMotion.current
    val amount by animateFloatAsState(if (checked) 1f else 0f,
        if (reduced) snap() else CLMotion.micro, label = "switch position")
    val colors = MaterialTheme.colorScheme
    val track by animateColorAsState(if (checked) colors.primary else colors.surfaceVariant,
        if (reduced) tween(180) else spring(dampingRatio = CLMotion.micro.dampingRatio,
            stiffness = CLMotion.micro.stiffness), label = "switch colour")
    val outline = colors.onSurfaceVariant
    val thumb = if (checked) colors.onPrimary else colors.onSurfaceVariant
    Canvas(Modifier.size(52.dp, 32.dp).graphicsLayer { alpha = if (enabled) 1f else 0.5f }) {
        drawRoundRect(track, cornerRadius = CornerRadius(16.dp.toPx()))
        if (!checked) drawRoundRect(outline, cornerRadius = CornerRadius(16.dp.toPx()), style = Stroke(2.dp.toPx()))
        drawCircle(thumb, 12.dp.toPx(), Offset((16.dp + 20.dp * amount).toPx(), size.height / 2))
    }
}

@Composable
fun CLChoiceChip(text: String, selected: Boolean, onClick: () -> Unit) {
    val source = remember { MutableInteractionSource() }
    val colors = MaterialTheme.colorScheme
    val color by animateColorAsState(if (selected) colors.primaryContainer else colors.surfaceVariant,
        if (LocalReduceMotion.current) tween(180) else spring(dampingRatio = CLMotion.micro.dampingRatio,
            stiffness = CLMotion.micro.stiffness), label = "choice colour")
    Box(Modifier.heightIn(min = 48.dp).clPress(source, LocalReduceMotion.current)
        .clip(RoundedCornerShape(CLTokens.chipRadius)).background(color)
        .selectable(selected, interactionSource = source, indication = null, role = Role.Tab, onClick = onClick)
        .padding(horizontal = 12.dp, vertical = 8.dp), contentAlignment = Alignment.Center) {
        Text(text, color = colors.onSurface, style = MaterialTheme.typography.labelMedium,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium)
    }
}

@Composable
fun CLEmptyState(title: String, body: String) = CLCard(Modifier.fillMaxWidth()) {
    Text(title, style = MaterialTheme.typography.headlineMedium)
    Text(body, style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
}
@Composable
fun CLStepList(steps: List<String>, current: Int) {
    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        steps.forEachIndexed { index, step ->
            val description = stringResource(R.string.cl_step_description, index + 1, steps.size, step)
            val status = stringResource(when {
                index < current -> R.string.cl_step_done
                index == current -> R.string.cl_step_current
                else -> R.string.cl_step_waiting
            })
            Row(Modifier.clearAndSetSemantics { contentDescription = description; stateDescription = status },
                horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(if (index < current) "✓" else "${index + 1}", color = MaterialTheme.colorScheme.primary)
                Text(step, style = MaterialTheme.typography.bodyLarge)
            }
        }
    }
}
@Composable
fun CLSourceChip(label: String, onClick: () -> Unit) = CLSecondaryButton(label, onClick)

@Composable
fun CLEmergencyStrip(text: String, action: String, onClick: () -> Unit) {
    CLCard(Modifier.fillMaxWidth()) {
        Text(text, style = MaterialTheme.typography.titleLarge)
        CLSecondaryButton(action, onClick)
    }
}

@Composable
fun Modifier.clMaterial(hazeState: HazeState, radius: Dp = 24.dp): Modifier {
    val solid = LocalReduceTransparency.current || Build.VERSION.SDK_INT < 31
    val color = MaterialTheme.colorScheme.surface
    return then(if (solid) Modifier.background(if (LocalReduceTransparency.current) color else color.copy(alpha = 0.95f))
            else Modifier.hazeEffect(state = hazeState) {
                backgroundColor = color
                blurRadius = radius
                tints = listOf(HazeTint(color.copy(alpha = 0.85f)))
            })
}

@Composable
fun CLGlass(hazeState: HazeState, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Box(modifier.clip(RoundedCornerShape(24.dp)).clMaterial(hazeState).padding(16.dp)) { content() }
}

val LocalNavigationContentPadding = staticCompositionLocalOf { PaddingValues() }
internal val LocalUnobscuredBringIntoViewSpec = staticCompositionLocalOf<BringIntoViewSpec?> { null }

@Composable
fun CLHorizontalChoices(content: @Composable RowScope.() -> Unit) {
    val spec = LocalUnobscuredBringIntoViewSpec.current ?: LocalBringIntoViewSpec.current
    CompositionLocalProvider(LocalBringIntoViewSpec provides spec) {
        Row(Modifier.horizontalScroll(rememberScrollState()).selectableGroup(),
            horizontalArrangement = Arrangement.spacedBy(8.dp), content = content)
    }
}

@Composable
fun CLLargeTitleScaffold(
    title: String,
    modifier: Modifier = Modifier,
    navigation: (@Composable () -> Unit)? = null,
    expandedTitle: (@Composable () -> Unit)? = null,
    maxContentWidth: Dp = 640.dp,
    sideMargin: Dp = 16.dp,
    content: @Composable ColumnScope.() -> Unit,
) {
    val scroll = rememberScrollState()
    val haze = rememberHazeState()
    val navigationPadding = LocalNavigationContentPadding.current
    val direction = LocalLayoutDirection.current
    var titleHeight by remember { mutableIntStateOf(1) }
    val progress by remember { derivedStateOf { (scroll.value.toFloat() / titleHeight).coerceIn(0f, 1f) } }
    val reduceMotion = LocalReduceMotion.current
    val solid = LocalReduceTransparency.current
    val fullWindowSpec = LocalUnobscuredBringIntoViewSpec.current ?: LocalBringIntoViewSpec.current
    val density = LocalDensity.current
    Scaffold(modifier = modifier, containerColor = Color.Transparent,
        contentWindowInsets = WindowInsets(0, 0, 0, 0),
        topBar = {
            Box(Modifier.fillMaxWidth().padding(
                start = navigationPadding.calculateStartPadding(direction),
                end = navigationPadding.calculateEndPadding(direction))) {
                if (progress > 0f) Box(Modifier.matchParentSize().graphicsLayer {
                    alpha = if (solid) 1f else progress
                    scaleX = if (reduceMotion) 1f else 0.96f + 0.04f * progress
                    scaleY = scaleX
                }.clip(RoundedCornerShape(bottomStart = 16.dp, bottomEnd = 16.dp))
                    .clMaterial(haze, 24.dp * progress))
                Row(Modifier.align(Alignment.Center).widthIn(max = 640.dp).fillMaxWidth()
                    .heightIn(min = 56.dp).padding(horizontal = 8.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    navigation?.invoke()
                    Text(title, Modifier.weight(1f).graphicsLayer { alpha = progress }
                        .then(if (progress < 0.5f) Modifier.clearAndSetSemantics {} else Modifier.semantics { heading() }),
                        style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold, letterSpacing = 0.01.em))
                }
            }
        }) { headerPadding ->
        val top = with(density) { headerPadding.calculateTopPadding().toPx() }
        val bottom = with(density) { navigationPadding.calculateBottomPadding().toPx() }
        val focusSpec = remember(fullWindowSpec, top, bottom) {
            object : BringIntoViewSpec {
                override fun calculateScrollDistance(offset: Float, size: Float, containerSize: Float) =
                    fullWindowSpec.calculateScrollDistance(offset - top, size, (containerSize - top - bottom).coerceAtLeast(1f))
            }
        }
        CompositionLocalProvider(LocalBringIntoViewSpec provides focusSpec,
            LocalUnobscuredBringIntoViewSpec provides fullWindowSpec) {
        Column(Modifier.fillMaxSize().hazeSource(haze).verticalScroll(scroll)
            .padding(navigationPadding).padding(top = if (navigation == null) 0.dp else headerPadding.calculateTopPadding())
            .padding(horizontal = sideMargin, vertical = 24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Column(Modifier.widthIn(max = maxContentWidth).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(24.dp)) {
                Box(Modifier.onSizeChanged { titleHeight = it.height.coerceAtLeast(1) }
                    .graphicsLayer { alpha = 1f - progress }
                    .then(if (progress >= 0.5f) Modifier.clearAndSetSemantics {} else Modifier.semantics(mergeDescendants = true) { heading() })) {
                    if (expandedTitle == null) Text(title, style = MaterialTheme.typography.displayLarge)
                    else expandedTitle()
                }
                content()
            }
        }
        }
    }
}
