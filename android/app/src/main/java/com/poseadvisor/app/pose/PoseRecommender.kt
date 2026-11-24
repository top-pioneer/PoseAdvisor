package com.poseadvisor.app.pose

import com.poseadvisor.app.scene.SceneClassifier

/**
 * 智能姿势推荐引擎：
 * 将场景多维特征与每个姿势的几何特征做亲和度打分，自动排序出
 * 与当前环境气质最契合的姿势候选队列，无需人工挑选。
 *
 * 匹配直觉：
 *  - 天空/开阔 → 双臂展开、腿部分开的舒展姿势
 *  - 绿地/自然 → 站立舒展、躯干略有动势的姿势
 *  - 弱光/室内 → 收敛、低重心姿势
 *  - 暖色氛围 → 适度抬手的柔和姿势
 */
object PoseRecommender {

    data class Ranked(
        val entry: PoseEntry,
        val features: PoseFeatures.Features,
        val affinity: Float,
    )

    fun rank(library: PoseLibrary, scene: SceneClassifier.SceneProfile): List<Ranked> {
        return library.poses.map { entry ->
            val f = PoseFeatures.of(entry.joints)
            Ranked(entry, f, affinity(f, scene))
        }.sortedByDescending { it.affinity }
    }

    private fun affinity(f: PoseFeatures.Features, s: SceneClassifier.SceneProfile): Float {
        var score = 0f
        // 开阔天空 ↔ 展开度
        score += s.skyRatio * (0.9f * f.spread + 0.6f * f.legSpread)
        // 自然绿意 ↔ 站立动势（腿部分开 + 适度躯干倾斜）
        score += s.greenRatio * (0.7f * f.legSpread + 0.8f * f.torsoTilt)
        // 弱光 ↔ 收敛低重心（收敛度越小越好）
        score += (1f - s.brightness).coerceIn(0f, 1f) * (1.2f - f.compactness).coerceIn(-0.5f, 1.2f)
        // 暖色氛围 ↔ 适度抬手（过度张扬反而减分）
        score += s.warmRatio * (if (f.armRaise in 0.15f..0.9f) 0.6f else 0.15f)
        // 画面活力 ↔ 整体展开
        score += s.colorfulness * 0.4f * f.spread
        // 弱光环境惩罚大倾斜（不稳）
        if (s.brightness < 0.33f) score -= 0.5f * f.torsoTilt
        return score
    }
}
