package com.poseadvisor.app.pose

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * 姿势几何特征向量：用于"场景 × 姿势"智能匹配。
 * 全部在归一化空间（骨盆原点、躯干单位长）计算，与位置/身高无关。
 */
object PoseFeatures {

    data class Features(
        val armRaise: Float,      // 抬手高度（肩到最高手腕的垂直距离，越高越张扬）
        val spread: Float,        // 双臂展开度（腕距/肩距）
        val legSpread: Float,     // 双腿展开度（踝距/肩距）
        val torsoTilt: Float,     // 躯干倾斜角（弧度，0=直立，越大越动感/随性）
        val compactness: Float,   // 肢体收敛度（各关节到骨盆的平均距离，越小越收敛）
    )

    fun of(pose75: FloatArray): Features {
        val p = PoseComparator.normalize(pose75)
        val shoulderY = (p.jy(PoseModel.SHOULDER_L) + p.jy(PoseModel.SHOULDER_R)) / 2
        val wristMinY = minOf(p.jy(PoseModel.WRIST_L), p.jy(PoseModel.WRIST_R))
        val shoulderSpan = abs(p.jx(PoseModel.SHOULDER_L) - p.jx(PoseModel.SHOULDER_R)) + 1e-6f

        val spread = abs(p.jx(PoseModel.WRIST_L) - p.jx(PoseModel.WRIST_R)) / shoulderSpan
        val legSpread = abs(p.jx(PoseModel.ANKLE_L) - p.jx(PoseModel.ANKLE_R)) / shoulderSpan

        // 躯干与竖直方向的夹角
        val dx = p.jx(PoseModel.NECK) - p.jx(PoseModel.PELVIS)
        val dy = p.jy(PoseModel.NECK) - p.jy(PoseModel.PELVIS)
        val tilt = abs(atan2(dx, -dy))

        var acc = 0f
        var cnt = 0
        for (i in intArrayOf(5, 16, 17, 7, 8, 13, 14)) {
            val d = sqrt(p.jx(i) * p.jx(i) + p.jy(i) * p.jy(i))
            acc += d; cnt++
        }

        return Features(
            armRaise = shoulderY - wristMinY,
            spread = spread,
            legSpread = legSpread,
            torsoTilt = tilt,
            compactness = acc / cnt,
        )
    }
}
