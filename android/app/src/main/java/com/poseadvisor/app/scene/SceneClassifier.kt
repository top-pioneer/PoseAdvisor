package com.poseadvisor.app.scene

import com.poseadvisor.app.camera.FrameBuffer

/**
 * 轻量场景识别：对最新预览帧做亚采样统计（亮度/绿色占比/天空占比），
 * 毫秒级完成，为姿势推荐选择默认类别。用户可随时手动切换类别。
 */
object SceneClassifier {

    data class SceneHint(val label: String, val suggestedCategory: String)

    fun classify(frame: FrameBuffer.Frame): SceneHint {
        val data = frame.nv21
        val w = frame.width
        val h = frame.height

        var brightSum = 0L
        var greenCount = 0
        var skyCount = 0
        var total = 0
        var topTotal = 0

        // 亚采样 Y 平面（每 8 行每 8 列取一个点）
        var r = 0
        while (r < h) {
            var c = 0
            while (c < w) {
                val y = data[r * w + c].toInt() and 0xFF
                brightSum += y
                // 用 UV 粗判绿色/蓝色（NV21：VU 交错）
                val uvIndex = w * h + (r / 2) * w + (c / 2) * 2
                if (uvIndex + 1 < data.size) {
                    val v = data[uvIndex].toInt() and 0xFF
                    val u = data[uvIndex + 1].toInt() and 0xFF
                    // 绿色：U、V 均偏低；天空蓝：U 高 V 中低
                    if (u < 100 && v < 110 && y in 40..220) greenCount++
                    if (r < h / 3) {
                        topTotal++
                        if (u > 130 && v < 130 && y > 80) skyCount++
                    }
                }
                total++
                c += 8
            }
            r += 8
        }

        if (total == 0) return SceneHint("日常街拍", "standing")
        val brightness = brightSum.toFloat() / total / 255f
        val greenRatio = greenCount.toFloat() / total
        val skyRatio = if (topTotal > 0) skyCount.toFloat() / topTotal else 0f

        return when {
            greenRatio > 0.25f -> SceneHint("户外自然", "standing")
            skyRatio > 0.30f -> SceneHint("开阔天空", "dynamic")
            brightness < 0.33f -> SceneHint("室内弱光", "sitting")
            brightness > 0.72f -> SceneHint("明亮通透", "gesture")
            else -> SceneHint("日常街拍", "standing")
        }
    }
}
