package com.poseadvisor.app.pose

/**
 * 与训练管线一致的 25 关节人体姿态表示。
 * 坐标系：MediaPipe 归一化图像坐标（x 右，y 下，z 深度），每关节 3 个浮点。
 */
object PoseModel {
    const val NUM_JOINTS = 25
    const val POSE_DIM = NUM_JOINTS * 3

    // 关节语义索引（与 mediapipe_to_pose_vector.py 训练格式一致）
    const val PELVIS = 0
    const val SPINE_LOWER = 1
    const val SPINE_MID = 2
    const val CHEST = 3
    const val NECK = 4
    const val HEAD = 5
    const val ANKLE_L = 7
    const val ANKLE_R = 8
    const val SHOULDER_L = 10
    const val SHOULDER_R = 11
    const val HIP_L = 12
    const val ELBOW_L = 13
    const val ELBOW_R = 14
    const val HIP_R = 15
    const val WRIST_L = 16
    const val WRIST_R = 17
    const val HAND_L = 18
    const val HAND_R = 19
    const val FOOT = 20
    const val NOSE = 22

    /** App 端渲染使用的解剖学骨架连接（剔除训练格式中的退化腿连接） */
    val RENDER_EDGES = listOf(
        0 to 1, 1 to 2, 2 to 3, 3 to 4, 4 to 5,
        3 to 10, 10 to 13, 13 to 16, 16 to 18,
        3 to 11, 11 to 14, 14 to 17, 17 to 19,
        0 to 12, 12 to 7,
        0 to 15, 15 to 8,
        5 to 22,
    )

    /** 用于姿势比对的关键肢体段（索引对 + 中文部位名 + 权重） */
    data class LimbSegment(val a: Int, val b: Int, val partName: String, val weight: Float)

    val COMPARE_SEGMENTS = listOf(
        LimbSegment(PELVIS, NECK, "躯干", 1.3f),
        LimbSegment(SHOULDER_L, ELBOW_L, "左上臂", 1.0f),
        LimbSegment(ELBOW_L, WRIST_L, "左前臂", 0.9f),
        LimbSegment(SHOULDER_R, ELBOW_R, "右上臂", 1.0f),
        LimbSegment(ELBOW_R, WRIST_R, "右前臂", 0.9f),
        LimbSegment(HIP_L, ANKLE_L, "左腿", 0.8f),
        LimbSegment(HIP_R, ANKLE_R, "右腿", 0.8f),
    )
}

/** 读取第 [idx] 个关节的 (x, y)，忽略 z。 */
fun FloatArray.jx(idx: Int) = this[idx * 3]
fun FloatArray.jy(idx: Int) = this[idx * 3 + 1]
