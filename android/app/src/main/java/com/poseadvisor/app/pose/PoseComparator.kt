package com.poseadvisor.app.pose

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * 姿势比对引擎：
 * 将用户实时姿态与推荐引导姿态做尺度/位置无关的归一化后，
 * 按肢体段方向角误差加权计算匹配度（0-100），并给出最大偏差部位的纠正提示。
 */
object PoseComparator {

    data class CompareResult(
        val score: Int,                       // 0-100 匹配度
        val segmentOk: BooleanArray,          // 每肢体段是否达标（用于着色）
        val hint: String?,                    // 最需要纠正的部位提示
    )

    private const val OK_THRESHOLD_RAD = 0.28f   // 单段达标阈值（约16°）
    private const val SCORE_K = 2.6f             // 误差→分数的衰减系数

    /** 归一化：骨盆平移到原点，按躯干长度（骨盆→颈）缩放，消除距离与身高差异 */
    fun normalize(pose: FloatArray): FloatArray {
        val out = pose.copyOf()
        val px = pose.jx(PoseModel.PELVIS)
        val py = pose.jy(PoseModel.PELVIS)
        val torso = sqrt(
            (pose.jx(PoseModel.NECK) - px) * (pose.jx(PoseModel.NECK) - px) +
                (pose.jy(PoseModel.NECK) - py) * (pose.jy(PoseModel.NECK) - py)
        ).coerceAtLeast(1e-4f)
        for (i in 0 until PoseModel.NUM_JOINTS) {
            out[i * 3] = (pose[i * 3] - px) / torso
            out[i * 3 + 1] = (pose[i * 3 + 1] - py) / torso
        }
        return out
    }

    /** 生成用于屏幕展示的引导骨架：骨盆置于 (0.5, 0.55)，躯干高度约 0.30 */
    fun guideForDisplay(guide: FloatArray): FloatArray {
        val n = normalize(guide)
        val scale = 0.30f
        for (i in 0 until PoseModel.NUM_JOINTS) {
            n[i * 3] = n[i * 3] * scale + 0.5f
            n[i * 3 + 1] = n[i * 3 + 1] * scale + 0.55f
        }
        return n
    }

    fun compare(userPose: FloatArray, guidePose: FloatArray): CompareResult {
        val u = normalize(userPose)
        val g = normalize(guidePose)

        val segs = PoseModel.COMPARE_SEGMENTS
        val segOk = BooleanArray(segs.size)
        var weightedSum = 0f
        var weightTotal = 0f
        var worstIdx = -1
        var worstErr = -1f

        for (i in segs.indices) {
            val s = segs[i]
            val angU = atan2(u.jy(s.b) - u.jy(s.a), u.jx(s.b) - u.jx(s.a))
            val angG = atan2(g.jy(s.b) - g.jy(s.a), g.jx(s.b) - g.jx(s.a))
            var err = abs(angU - angG)
            if (err > Math.PI.toFloat()) err = (2 * Math.PI).toFloat() - err
            segOk[i] = err < OK_THRESHOLD_RAD
            weightedSum += err * s.weight
            weightTotal += s.weight
            if (err > worstErr) { worstErr = err; worstIdx = i }
        }

        val mean = weightedSum / weightTotal
        val score = (100f * exp(-SCORE_K * mean)).toInt().coerceIn(0, 100)

        val hint = if (worstIdx >= 0 && worstErr > OK_THRESHOLD_RAD) {
            val s = segs[worstIdx]
            // 用远端关节的相对位移给出方向提示
            val dx = g.jx(s.b) - u.jx(s.b)
            val dy = g.jy(s.b) - u.jy(s.b)
            val dir = if (abs(dy) >= abs(dx)) {
                if (dy < 0) "上" else "下"
            } else {
                if (dx < 0) "左" else "右"
            }
            "请把${s.partName}向${dir}调整"
        } else null

        return CompareResult(score, segOk, hint)
    }
}
