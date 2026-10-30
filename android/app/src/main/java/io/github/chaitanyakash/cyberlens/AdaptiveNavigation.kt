package io.github.chaitanyakash.cyberlens

import androidx.compose.foundation.*
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.material3.*
import androidx.compose.material3.adaptive.currentWindowAdaptiveInfo
import androidx.compose.material3.adaptive.navigationsuite.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.semantics.*
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.layout.layout
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInWindow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.draw.clipToBounds
import kotlin.math.roundToInt
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.*
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import kotlin.math.ceil
import dev.chrisbanes.haze.*
import io.github.chaitanyakash.cyberlens.core.designsystem.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun AdaptiveNavigation(
    selected: Destination,
    onSelect: (Destination) -> Unit,
    content: @Composable () -> Unit,
) {
    val haze = rememberHazeState()
    val haptics = LocalHapticFeedback.current
    val measurer = rememberTextMeasurer()
    val density = LocalDensity.current
    var focusedDestination by remember { mutableStateOf<Destination?>(null) }
    var origin by remember { mutableStateOf(Offset.Zero) }
    BoxWithConstraints(Modifier.fillMaxSize().onGloballyPositioned { origin = it.positionInWindow() }) {
        val windowWidth = maxWidth
        val windowHeight = maxHeight
        val labels = Destination.entries.map { stringResource(it.label) }
        val labelStyle = MaterialTheme.typography.labelMedium
        val labelWidth = labels.maxOf { measurer.measure(it, labelStyle, softWrap = false).size.width }
        val compactRailWidth = (ceil((with(density) { labelWidth.toDp().value } + 24f) / 4f) * 4f).dp.coerceAtLeast(96.dp)
        val wordmarkWidth = measurer.measure("CyberLens", TextStyle(fontFamily = CLWordmarkFont,
            fontSize = 28.sp, fontWeight = FontWeight.SemiBold, letterSpacing = (-0.01).em)).size.width
        val expandedRailWidth = (ceil((with(density) { wordmarkWidth.toDp().value } + 64f) / 4f) * 4f).dp.coerceAtLeast(260.dp)
        val posture = LocalPostureOverride.current ?: currentWindowAdaptiveInfo().windowPosture
        val regions = hingeRegions(with(density) { windowWidth.toPx() },
            with(density) { windowHeight.toPx() }, origin, posture)
        val type = when {
            regions != null && !regions.vertical -> NavigationSuiteType.ShortNavigationBarMedium
            regions != null && regions.vertical -> if (with(density) { regions.navigation.width.toDp() } >= 260.dp)
                NavigationSuiteType.WideNavigationRailExpanded else NavigationSuiteType.WideNavigationRailCollapsed
            windowHeight < 480.dp -> NavigationSuiteType.ShortNavigationBarMedium
            windowWidth >= 1200.dp -> NavigationSuiteType.WideNavigationRailExpanded
            posture.isTabletop -> NavigationSuiteType.ShortNavigationBarMedium
            windowWidth >= 600.dp -> NavigationSuiteType.WideNavigationRailCollapsed
            else -> NavigationSuiteType.ShortNavigationBarCompact
        }
        val horizontal = type == NavigationSuiteType.ShortNavigationBarCompact ||
            type == NavigationSuiteType.ShortNavigationBarMedium
        val navigation: @Composable () -> Unit = {
            val reservation = if (regions == null) Modifier else Modifier.layout { measurable, constraints ->
                val bounds = regions.navigation
                val child = measurable.measure(Constraints.fixed(bounds.width.roundToInt(), bounds.height.roundToInt()))
                // Reserve the crease as well, but never draw or put targets in it.
                if (regions.vertical) layout(regions.content.left.roundToInt(), constraints.maxHeight) {
                    child.place(0, 0)
                } else layout(constraints.maxWidth, (windowHeight.toPx() - regions.content.bottom).roundToInt()) {
                    child.place(0, (bounds.top - regions.content.bottom).roundToInt())
                }
            }
            Box(reservation) {
                val safe = WindowInsets.safeDrawing.asPaddingValues()
                val direction = LocalLayoutDirection.current
                val expanded = type == NavigationSuiteType.WideNavigationRailExpanded
                val navigationModifier = if (regions != null) Modifier.fillMaxSize() else if (horizontal) Modifier.fillMaxWidth() else
                    Modifier.width((if (expanded) expandedRailWidth else compactRailWidth) + safe.calculateStartPadding(direction)).fillMaxHeight()
                Box(navigationModifier.testTag("navigation").selectableGroup().clMaterial(haze)) {
                    if (horizontal) {
                        val available = with(density) {
                            (windowWidth - safe.calculateStartPadding(direction) - safe.calculateEndPadding(direction)).toPx() /
                                Destination.entries.size - 8.dp.toPx()
                        }
                        val sharedLabel = labelWidth > available
                        Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth()
                            .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Horizontal + WindowInsetsSides.Bottom))) {
                            if (sharedLabel) Text(stringResource((focusedDestination ?: selected).label),
                                Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp)
                                    .semantics { hideFromAccessibility() }.testTag("navigation_label"),
                                style = labelStyle.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.onSurface, textAlign = TextAlign.Center,
                                maxLines = 1, softWrap = false)
                            Row(Modifier.fillMaxWidth().height(IntrinsicSize.Min)) {
                                Destination.entries.forEach { destination ->
                                    NavigationDestination(destination, selected == destination, false,
                                        iconOnly = sharedLabel,
                                        modifier = Modifier.weight(1f).fillMaxHeight().onFocusChanged {
                                            if (it.isFocused) focusedDestination = destination
                                            else if (focusedDestination == destination) focusedDestination = null
                                        }, onClick = {
                                            focusedDestination = null
                                            haptics.performHapticFeedback(HapticFeedbackType.SegmentTick)
                                            onSelect(destination)
                                        })
                                }
                            }
                        }
                    } else {
                        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())
                            .heightIn(min = windowHeight)
                            .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Vertical + WindowInsetsSides.Start))
                            .padding(horizontal = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            if (expanded) Box(Modifier.padding(24.dp)) { CLWordmark() }
                            Destination.entries.forEach { destination ->
                                NavigationDestination(destination, selected == destination, expanded,
                                    modifier = Modifier.fillMaxWidth(), onClick = {
                                        haptics.performHapticFeedback(HapticFeedbackType.SegmentTick)
                                        onSelect(destination)
                                    })
                            }
                        }
                    }
                }
        }
        }
        CompositionLocalProvider(LocalModalPaneBounds provides regions?.sheet?.translate(origin.x, origin.y)) {
        NavigationSuiteScaffoldLayout(navigationSuiteType = type, navigationSuite = navigation) {
            BoxWithConstraints {
                val safe = WindowInsets.safeDrawing.asPaddingValues()
                val direction = LocalLayoutDirection.current
                val navigationPadding = if (regions != null) PaddingValues() else PaddingValues(
                    start = (windowWidth - maxWidth - safe.calculateStartPadding(direction)).coerceAtLeast(0.dp),
                    bottom = (windowHeight - maxHeight - safe.calculateBottomPadding()).coerceAtLeast(0.dp),
                )
                CompositionLocalProvider(LocalNavigationContentPadding provides navigationPadding) {
                    Box(Modifier.testTag("pane_content").layout { measurable, constraints ->
                        val width = regions?.content?.width?.roundToInt() ?: windowWidth.roundToPx()
                        val height = regions?.content?.height?.roundToInt() ?: windowHeight.roundToPx()
                        val placeable = measurable.measure(Constraints.fixed(width, height))
                        layout(constraints.maxWidth, constraints.maxHeight) {
                            placeable.placeRelative(if (regions == null) -(width - constraints.maxWidth) else 0, 0)
                        }
                    }.then(if (regions != null) Modifier.clipToBounds() else Modifier)
                        .hazeSource(haze).windowInsetsPadding(WindowInsets.safeDrawing)) { content() }
                }
            }
        }
        }
    }
}

@Composable
private fun NavigationDestination(destination: Destination, selected: Boolean, expanded: Boolean,
    iconOnly: Boolean = false, modifier: Modifier = Modifier, onClick: () -> Unit) {
    val interactions = remember { MutableInteractionSource() }
    val reduced = LocalReduceMotion.current
    val selection by animateFloatAsState(if (selected) 1f else 0f,
        if (reduced) tween(180) else CLMotion.micro, label = "navigation selection")
    val colors = MaterialTheme.colorScheme
    val tint = if (selected) colors.primary else colors.onSurfaceVariant
    val target = modifier.clPress(interactions, reduced)
        .selectable(selected, interactionSource = interactions, indication = null, role = Role.Tab, onClick = onClick)
    if (expanded) {
        Box(target.fillMaxWidth().heightIn(min = 56.dp)) {
            Box(Modifier.matchParentSize().graphicsLayer {
                alpha = selection
                scaleX = if (reduced) 1f else 0.96f + 0.04f * selection
                scaleY = scaleX
            }.background(colors.primaryContainer, RoundedCornerShape(16.dp)))
            Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Icon(destination.icon, null, Modifier.size(24.dp), tint = tint)
                Text(stringResource(destination.label), Modifier.weight(1f), color = tint,
                    style = MaterialTheme.typography.titleMedium)
            }
        }
    } else {
        Column(target.fillMaxWidth().heightIn(min = if (iconOnly) 56.dp else 72.dp).padding(vertical = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Box(Modifier.size(56.dp, 40.dp), contentAlignment = Alignment.Center) {
                Box(Modifier.matchParentSize().graphicsLayer {
                    alpha = selection
                    scaleX = if (reduced) 1f else 0.8f + 0.2f * selection
                    scaleY = scaleX
                }.background(colors.primaryContainer, RoundedCornerShape(50)))
                Icon(destination.icon, if (iconOnly) stringResource(destination.label) else null, Modifier.size(24.dp), tint = tint)
            }
            if (!iconOnly) Text(stringResource(destination.label), color = tint, textAlign = TextAlign.Center,
                style = MaterialTheme.typography.labelMedium, maxLines = 1, softWrap = false)
        }
    }
}
