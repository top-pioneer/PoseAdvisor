package com.poseadvisor.app.pose

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * 端上姿势合成器：在库中最契合的两个姿势之间插值，并施加小扰动，
 * 再经"自然性投影"（肢体长度锁定 + 关节角度生物力学限幅），
 * 创造出姿势库中不存在、但解剖学上自然可信的新姿势。
 *
 * 自然性优先：任何不满足约束的结果自动回退为纯插值姿势。
 */
object PoseSynthesizer {

    /** 肢体段（两端关节索引 + 中文名）——长度需要锁定 */
    private val BONES = listOf(
        PoseModel.PELVIS to PoseModel.NECK,          // 躯干
        PoseModel.NECK to PoseModel.HEAD,            // 颈头
        PoseModel.SHOULDER_L to PoseModel.ELBOW_L,   // 左上臂
        PoseModel.ELBOW_L to PoseModel.WRIST_L,      // 左前臂
        PoseModel.SHOULDER_R to PoseModel.ELBOW_R,   // 右上臂
        PoseModel.ELBOW_R to PoseModel.WRIST_R,      // 右前臂
        PoseModel.HIP_L to PoseModel.ANKLE_L,        // 左腿
        PoseModel.HIP_R to PoseModel.ANKLE_R,        // 右腿
        PoseModel.SHOULDER_L to PoseModel.SHOULDER_R, // 肩宽
        PoseModel.HIP_L to PoseModel.HIP_R,           // 髋宽
    )

    /** 关节角度限幅（弧度）：[顶点, 近端, 远端, 最小角, 最大角] */
    private data class AngleLimit(val apex: Int, val from: Int, val to: Int, val minDeg: Float, val maxDeg: Float)

    private val ANGLE_LIMITS = listOf(
        AngleLimit(PoseModel.ELBOW_L, PoseModel.SHOULDER_L, PoseModel.WRIST_L, 12f, 178f),
        AngleLimit(PoseModel.ELBOW_R, PoseModel.SHOULDER_R, PoseModel.WRIST_R, 12f, 178f),
        AngleLimit(PoseModel.SHOULDER_L, PoseModel.NECK, PoseModel.ELBOW_L, 8f, 170f),
        AngleLimit(PoseModel.SHOULDER_R, PoseModel.NECK, PoseModel.ELBOW_R, 8f, 170f),
        AngleLimit(PoseModel.HIP_L, PoseModel.PELVIS, PoseModel.ANKLE_L, 5f, 172f),
        AngleLimit(PoseModel.HIP_R, PoseModel.PELVIS, PoseModel.ANKLE_R, 5f, 172f),
    )

    /**
     * 合成一个新姿势。
     * @param a, b 两个种子姿势（75维，归一化或原始均可，内部统一归一化）
     * @param lambda 插值系数（0~1，默认随机 0.3~0.7）
     * @param noise 扰动强度（躯干单位，默认 0.05）
     */
    fun synthesize(
        a: FloatArray,
        b: FloatArray,
        lambda: Float = 0.3f + Random.nextFloat() * 0.4f,
        noise: Float = 0.05f,
    ): FloatArray {
        val na = PoseComparator.normalize(a)
        val nb = PoseComparator.normalize(b)

        // 1. 插值
        val interp = FloatArray(PoseModel.POSE_DIM) { i -> na[i] * lambda + nb[i] * (1f - lambda) }

        // 2. 小扰动（只动四肢末端与肘/腕/踝，躯干与头保持稳定 = 自然感来源之一）
        val perturbed = interp.copyOf()
        val movable = intArrayOf(13, 14, 16, 17, 18, 19, 7, 8)
        for (jt in movable) {
            perturbed[jt * 3] += (Random.nextFloat() - 0.5f) * 2f * noise
            perturbed[jt * 3 + 1] += (Random.nextFloat() - 0.5f) * 2f * noise
        }

        // 3. 自然性投影：锁定肢体长度 + 关节角度限幅（迭代 2 轮收敛）
        val natural = perturbed.copyOf()
        repeat(2) {
            projectBoneLengths(natural, reference = interp)
            projectAngleLimits(natural)
        }

        // 4. 验收：段长误差与角度均达标才返回合成结果，否则回退纯插值
        return if (isNatural(natural, interp)) natural else interp
    }

    /** 将每段骨骼的远端点沿当前方向重定位到参考段长 */
    private fun projectBoneLengths(pose: FloatArray, reference: FloatArray) {
        for ((ai, bi) in BONES) {
            val ax = pose.jx(ai); val ay = pose.jy(ai)
            var dx = pose.jx(bi) - ax; var dy = pose.jy(bi) - ay
            val len = sqrt(dx * dx + dy * dy)
            if (len < 1e-6f) continue
            val refLen = sqrt(
                (reference.jx(bi) - reference.jx(ai)).let { it * it } +
                    (reference.jy(bi) - reference.jy(ai)).let { it * it }
            )
            dx /= len; dy /= len
            pose[bi * 3] = ax + dx * refLen
            pose[bi * 3 + 1] = ay + dy * refLen
        }
    }

    /** 关节角度限幅：超过范围时把远端点旋转回边界角度 */
    private fun projectAngleLimits(pose: FloatArray) {
        for (lim in ANGLE_LIMITS) {
            val apexX = pose.jx(lim.apex); val apexY = pose.jy(lim.apex)
            val a1 = atan2(pose.jy(lim.from) - apexY, pose.jx(lim.from) - apexX)
            val a2 = atan2(pose.jy(lim.to) - apexY, pose.jx(lim.to) - apexX)
            var deg = Math.toDegrees((a2 - a1).toDouble()).toFloat()
            while (deg > 180f) deg -= 360f
            while (deg < -180f) deg += 360f
            val absDeg = kotlin.math.abs(deg)
            val clamped = absDeg.coerceIn(lim.minDeg, lim.maxDeg)
            if (clamped == absDeg) continue
            // 远端点重定位：保持与 apex 距离，角度贴到边界
            val sign = if (deg >= 0) 1f else -1f
            val newA2 = a1 + Math.toRadians((clamped * sign).toDouble()).toFloat()
            val rx = pose.jx(lim.to) - apexX
            val ry = pose.jy(lim.to) - apexY
            val r = sqrt(rx * rx + ry * ry)
            pose[lim.to * 3] = apexX + cos(newA2) * r
            pose[lim.to * 3 + 1] = apexY + sin(newA2) * r
        }
    }

    /** 自然性验收：骨骼长度误差 ≤8%，且头在骨盆上方、人体不翻转 */
    private fun isNatural(pose: FloatArray, reference: FloatArray): Boolean {
        for ((ai, bi) in BONES) {
            val len = dist(pose, ai, bi)
            val refLen = dist(reference, ai, bi)
            if (refLen > 1e-4f && kotlin.math.abs(len - refLen) / refLen > 0.08f) return false
        }
        // 头必须在骨盆上方（y 向下）
        if (pose.jy(PoseModel.HEAD) > pose.jy(PoseModel.PELVIS)) return false
        return true
    }

    private fun dist(p: FloatArray, a: Int, b: Int): Float {
        val dx = p.jx(b) - p.jx(a); val dy = p.jy(b) - p.jy(a)
        return sqrt(dx * dx + dy * dy)
    }
}
