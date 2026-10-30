package io.github.chaitanyakash.cyberlens

import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.*
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.platform.*
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.*
import androidx.compose.ui.tooling.preview.*
import androidx.compose.ui.unit.*
import dev.chrisbanes.haze.*
import io.github.chaitanyakash.cyberlens.core.designsystem.*

@Composable
fun ComponentCatalogue(onBack: () -> Unit = {}, initialDark: Boolean = false,
    initialSection: Int = 0, initialSheet: Boolean = false) {
    var dark by rememberSaveable { mutableStateOf(initialDark) }
    var large by rememberSaveable { mutableStateOf(false) }
    var motion by rememberSaveable { mutableStateOf(false) }
    var solid by rememberSaveable { mutableStateOf(false) }
    var section by rememberSaveable { mutableIntStateOf(initialSection.coerceIn(0, 3)) }
    val density = LocalDensity.current
    val inheritedLarge = density.fontScale >= 2f
    val catalogueHaptics = LocalHapticFeedback.current
    val inheritedMotion = LocalReduceMotion.current
    val inheritedSolid = LocalReduceTransparency.current
    val effectiveMotion = motion || inheritedMotion
    val effectiveSolid = solid || inheritedSolid
    val previewDensity = if (large && !inheritedLarge) Density(density.density, 2f) else density
    CompositionLocalProvider(LocalDensity provides previewDensity) {
        CyberLensTheme(dark = dark, reduceMotion = effectiveMotion, reduceTransparency = effectiveSolid) {
            val padding = LocalNavigationContentPadding.current
            val direction = LocalLayoutDirection.current
            val background = MaterialTheme.colorScheme.background
            Box(Modifier.fillMaxSize().drawBehind {
                val left = padding.calculateLeftPadding(direction).toPx()
                val right = padding.calculateRightPadding(direction).toPx()
                val top = padding.calculateTopPadding().toPx()
                val bottom = padding.calculateBottomPadding().toPx()
                drawRect(background, Offset(left, top), Size(
                    (size.width - left - right).coerceAtLeast(0f),
                    (size.height - top - bottom).coerceAtLeast(0f)))
            }) {
                CompositionLocalProvider(LocalContentColor provides MaterialTheme.colorScheme.onBackground) {
                    CLLargeTitleScaffold(stringResource(R.string.components),
                        navigation = { CLSecondaryButton(stringResource(R.string.back), onBack) }) {
                        Text(stringResource(R.string.catalogue_body), style = MaterialTheme.typography.bodyLarge)
                        CLGroupedList {
                            CLToggleRow(stringResource(R.string.dark_mode), dark, { dark = it })
                            CLToggleRow(stringResource(R.string.large_text), large || inheritedLarge, { large = it },
                                detail = if (inheritedLarge) stringResource(R.string.inherited_accessibility) else "", enabled = !inheritedLarge)
                            CLToggleRow(stringResource(R.string.reduce_motion), effectiveMotion, { motion = it },
                                detail = if (inheritedMotion) stringResource(R.string.inherited_accessibility) else "", enabled = !inheritedMotion)
                            CLToggleRow(stringResource(R.string.reduce_transparency), effectiveSolid, { solid = it },
                                detail = if (inheritedSolid) stringResource(R.string.inherited_accessibility) else "", enabled = !inheritedSolid)
                        }
                        CLHorizontalChoices {
                            listOf(R.string.fonts, R.string.buttons, R.string.cards, R.string.progress).forEachIndexed { index, label ->
                                CLChoiceChip(stringResource(label), selected = index == section, onClick = {
                                    if (section != index) {
                                        section = index
                                        catalogueHaptics.performHapticFeedback(androidx.compose.ui.hapticfeedback.HapticFeedbackType.SegmentTick)
                                    }
                                })
                            }
                        }
                        CatalogueSamples(section, initialSheet = initialSheet && section == 3)
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CatalogueSamples(section: Int, initialSheet: Boolean = false) {
    var sheet by rememberSaveable { mutableStateOf(initialSheet) }
    var selected by rememberSaveable { mutableStateOf(false) }
    var source by rememberSaveable { mutableStateOf(false) }
    val haptic = LocalHapticFeedback.current
    Column(Modifier.widthIn(max = 640.dp).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(24.dp)) {
        when(section) {
            0 -> {
                CLWordmark()
                Text(stringResource(R.string.setup_title), style = MaterialTheme.typography.displayLarge)
                Text(stringResource(R.string.sample_sheet), style = MaterialTheme.typography.headlineLarge)
                Text(stringResource(R.string.sample_verdict), style = MaterialTheme.typography.headlineMedium)
                Text(stringResource(R.string.sample_detail), style = MaterialTheme.typography.bodyLarge)
                Text(stringResource(R.string.sample_disclaimer), style = MaterialTheme.typography.labelMedium)
            }
            1 -> {
                CLPrimaryButton(stringResource(if (selected) R.string.sample_confirm else R.string.sample_button), {
                    selected = !selected
                    haptic.performHapticFeedback(androidx.compose.ui.hapticfeedback.HapticFeedbackType.ToggleOn)
                })
                CLSecondaryButton(stringResource(R.string.sample_secondary), { selected = !selected })
                CLChip(stringResource(R.string.sample_cue))
                CLSourceChip(stringResource(R.string.sample_source), { source = !source })
                if (source) Text(stringResource(R.string.source_hint))
            }
            2 -> {
                CLVerdictCard(stringResource(R.string.sample_verdict), stringResource(R.string.sample_detail))
                CLGroupedList { CLListRow(stringResource(R.string.sample_row), stringResource(R.string.sample_row_detail)) }
                CLEmptyState(stringResource(R.string.sample_empty_title), stringResource(R.string.sample_empty_body))
                GlassSample()
                CLEmergencyStrip(stringResource(R.string.emergency), stringResource(R.string.call_1930), {})
            }
            else -> {
                CLCard {
                    CLStepList(listOf(stringResource(R.string.sample_step1), stringResource(R.string.sample_step2),
                        stringResource(R.string.sample_step3)), 1)
                }
                CLPrimaryButton(stringResource(R.string.open_sheet), { sheet = true })
            }
        }
    }
    if (sheet) {
        CLQuestionSheet(title = stringResource(R.string.sample_sheet), onDismiss = { sheet = false }) { dismissSheet ->
            Text(stringResource(R.string.sample_question), style = MaterialTheme.typography.headlineMedium)
            listOf(R.string.yes, R.string.no, R.string.not_sure).forEach { option ->
                CLSecondaryButton(stringResource(option), {
                    haptic.performHapticFeedback(androidx.compose.ui.hapticfeedback.HapticFeedbackType.SegmentTick)
                    dismissSheet()
                })
            }
            CLSecondaryButton(stringResource(R.string.skip), dismissSheet)
        }
    }
}

@Composable
fun GlassSample() {
    val haze = rememberHazeState()
    Box(Modifier.fillMaxWidth().heightIn(min = 200.dp)) {
        Box(Modifier.fillMaxWidth().hazeSource(haze)
            .background(MaterialTheme.colorScheme.primaryContainer).padding(16.dp)) {
            Text(stringResource(R.string.sample_detail), style = MaterialTheme.typography.headlineMedium)
        }
        CLGlass(haze, Modifier.align(Alignment.BottomCenter).fillMaxWidth()) {
            Text(stringResource(R.string.component_demo))
        }
    }
}

@PreviewScreenSizes
@PreviewFontScale
@Composable
private fun ComponentsPreview() { ComponentCatalogue() }
