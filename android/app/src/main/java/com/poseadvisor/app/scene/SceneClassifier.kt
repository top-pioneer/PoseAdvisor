package com.poseadvisor.app.scene

import com.poseadvisor.app.camera.FrameBuffer
import kotlin.math.abs

/**
 * 轻量场景理解：对最新预览帧亚采样统计，输出多维场景特征向量，
 * 毫秒级完成。既给出人类可读的场景标签，也为姿势推荐引擎提供
 * 连续的特征输入（亮度/绿意/天空/暖色/画面活力）。
 */
object SceneClassifier {

    /** 多维场景特征（0~1 归一） */
    data class SceneProfile(
        val brightness: Float,    // 亮度
        val greenRatio: Float,    // 植被占比
        val skyRatio: Float,      // 天空占比（画面上部蓝色）
        val warmRatio: Float,     // 暖色占比（夕阳/暖光氛围）
        val colorfulness: Float,  // 画面活力（色彩丰富度）
    )

    data class SceneHint(val label: String, val profile: SceneProfile)

    fun classify(frame: FrameBuffer.Frame): SceneHint {
        val data = frame.nv21
        val w = frame.width
        val h = frame.height

        var brightSum = 0L
        var satSum = 0L
        var greenCount = 0
        var skyCount = 0
        var warmCount = 0
        var total = 0
        var topTotal = 0

        var r = 0
        while (r < h) {
            var c = 0
            while (c < w) {
                val y = data[r * w + c].toInt() and 0xFF
                brightSum += y
                val uvIndex = w * h + (r / 2) * w + (c / 2) * 2
                if (uvIndex + 1 < data.size) {
                    val v = data[uvIndex].toInt() and 0xFF      // V：红-青轴
                    val u = data[uvIndex + 1].toInt() and 0xFF  // U：蓝-黄轴
                    val saturation = abs(u - 128) + abs(v - 128)
                    satSum += saturation

                    if (u < 100 && v < 110 && y in 40..220) greenCount++
                    if (v > 145 && u < 135 && y > 60) warmCount++           // 偏红暖调
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

        if (total == 0) {
            return SceneHint("日常街拍", SceneProfile(0.5f, 0f, 0f, 0f, 0.3f))
        }
        val profile = SceneProfile(
            brightness = brightSum.toFloat() / total / 255f,
            greenRatio = greenCount.toFloat() / total,
            skyRatio = if (topTotal > 0) skyCount.toFloat() / topTotal else 0f,
            warmRatio = warmCount.toFloat() / total,
            colorfulness = (satSum.toFloat() / total / 128f).coerceIn(0f, 1f),
        )

        val label = when {
            profile.greenRatio > 0.25f -> "户外自然"
            profile.skyRatio > 0.30f -> "开阔天空"
            profile.warmRatio > 0.30f -> "暖调氛围"
            profile.brightness < 0.33f -> "室内弱光"
            profile.brightness > 0.72f -> "明亮通透"
            else -> "日常街拍"
        }
        return SceneHint(label, profile)
    }
}
