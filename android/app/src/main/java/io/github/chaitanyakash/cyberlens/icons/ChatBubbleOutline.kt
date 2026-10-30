/*
 * Copyright 2025 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.chaitanyakash.cyberlens.icons

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.materialIcon
import androidx.compose.material.icons.materialPath
import androidx.compose.ui.graphics.vector.ImageVector

public val Icons.Rounded.ChatBubbleOutline: ImageVector
    get() {
        if (_chatBubbleOutline != null) {
            return _chatBubbleOutline!!
        }
        _chatBubbleOutline = materialIcon(name = "Rounded.ChatBubbleOutline") {
            materialPath {
                moveTo(20.0f, 4.0f)
                verticalLineToRelative(12.0f)
                horizontalLineTo(5.17f)
                lineTo(4.0f, 17.17f)
                verticalLineTo(4.0f)
                horizontalLineTo(20.0f)
                moveTo(20.0f, 2.0f)
                horizontalLineTo(4.0f)
                curveTo(2.9f, 2.0f, 2.0f, 2.9f, 2.0f, 4.0f)
                verticalLineToRelative(15.59f)
                curveToRelative(0.0f, 0.89f, 1.08f, 1.34f, 1.71f, 0.71f)
                lineTo(6.0f, 18.0f)
                horizontalLineToRelative(14.0f)
                curveToRelative(1.1f, 0.0f, 2.0f, -0.9f, 2.0f, -2.0f)
                verticalLineTo(4.0f)
                curveTo(22.0f, 2.9f, 21.1f, 2.0f, 20.0f, 2.0f)
                lineTo(20.0f, 2.0f)
                close()
            }
        }
        return _chatBubbleOutline!!
    }

private var _chatBubbleOutline: ImageVector? = null
