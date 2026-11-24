package com.poseadvisor.app.pose

import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/**
 * 将 MediaPipe 33 个姿态关键点转换为与训练管线一致的 25 关节 75 维向量，
 * 逻辑与 PC 端 mediapipe_to_pose_vector.py 完全对齐。
 */
object PoseMapper {

    fun to75d(result: PoseLandmarkerResult): FloatArray? {
        if (result.landmarks().isEmpty()) return null
        val lms = result.landmarks()[0]
        if (lms.size < 33) return null

        val j = FloatArray(PoseModel.POSE_DIM)

        fun put(mpIdx: Int, ourIdx: Int) {
            val lm = lms[mpIdx]
            j[ourIdx * 3] = lm.x()
            j[ourIdx * 3 + 1] = lm.y()
            j[ourIdx * 3 + 2] = lm.z()
        }

        // 直接映射的关键点
        put(0, 22)   // 鼻子
        put(2, 23)   // 左眼
        put(5, 24)   // 右眼
        put(11, 10)  // 左肩
        put(12, 11)  // 右肩
        put(25, 12)  // 左髋
        put(26, 15)  // 右髋
        put(13, 13)  // 左肘
        put(14, 14)  // 右肘
        put(15, 16)  // 左腕
        put(16, 17)  // 右腕
        put(17, 18)  // 左手
        put(18, 19)  // 右手
        put(27, 4)   // 左膝（会被颈部覆盖，与训练格式一致）
        put(28, 5)   // 右膝（会被头部覆盖）
        put(29, 7)   // 左踝
        put(30, 8)   // 右踝
        put(31, 20)  // 左脚跟
        put(32, 20)  // 左脚趾

        // 脊柱点估计（与训练脚本相同：髋/肩平均）
        fun avg(dst: Int, vararg src: Int, extraY: Float = 0f) {
            var x = 0f; var y = 0f; var z = 0f
            for (s in src) { x += j.jx(s); y += j.jy(s); z += j[s * 3 + 2] }
            j[dst * 3] = x / src.size
            j[dst * 3 + 1] = y / src.size + extraY
            j[dst * 3 + 2] = z / src.size
        }

        avg(PoseModel.PELVIS, 12, 15)                    // 骨盆中心
        avg(2, 10, 11, 12, 15)                           // 脊柱中
        avg(3, 10, 11)                                   // 脊柱上（胸口）
        // 脊柱下：0.3 * 四点平均 + 0.7 * 骨盆
        j[1 * 3] = j.jx(2) * 0.3f + j.jx(0) * 0.7f
        j[1 * 3 + 1] = j.jy(2) * 0.3f + j.jy(0) * 0.7f
        j[1 * 3 + 2] = j[2 * 3 + 2] * 0.3f + j[0 * 3 + 2] * 0.7f
        // 颈部：双肩与鼻平均
        j[4 * 3] = (j.jx(10) + j.jx(11) + j.jx(22)) / 3f
        j[4 * 3 + 1] = (j.jy(10) + j.jy(11) + j.jy(22)) / 3f
        j[4 * 3 + 2] = (j[10 * 3 + 2] + j[11 * 3 + 2] + j[22 * 3 + 2]) / 3f
        // 头部：双眼中点略上移
        j[5 * 3] = (j.jx(23) + j.jx(24)) / 2f
        j[5 * 3 + 1] = (j.jy(23) + j.jy(24)) / 2f + 0.1f
        j[5 * 3 + 2] = (j[23 * 3 + 2] + j[24 * 3 + 2]) / 2f

        return j
    }
}
