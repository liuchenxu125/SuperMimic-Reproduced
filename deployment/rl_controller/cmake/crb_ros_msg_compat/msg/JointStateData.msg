# JointStateData.msg - 关节状态/命令反馈（含当前生效 kp/kd，不含灵巧手 kp/kd）

std_msgs/Header header

string[] name
float64[] position
float64[] velocity
float64[] effort
float64[] kp
float64[] kd
