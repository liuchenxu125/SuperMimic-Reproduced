
# Topic: joystick_events   qos:10  默认频率20HZ
# A:0;     B :1;    X : 2;    Y : 3;
# LB : 4;  RB : 5;  BACK : 6;   START : 7;
# LEFT_AXIS : 9;  //左摇杆按下    RIGHT_AXIS : 10; //右摇杆按下
# LT : 11;  RT : 12;


# 按钮事件数组（存储被操作的按钮ID）
uint32[] long_pressed    # 长按按钮集合
uint32[] single_clicked  # 单击按钮集合
uint32[] double_clicked  # 双击按钮集合

#十字按钮数据
int8 axis_x
int8 axis_y

# 摇杆数据（归一化到[-1.0, 1.0]）
#左摇杆的数据输出不是随着推杆推动而逐步递增
float32 left_x
float32 left_y
float32 right_x
float32 right_y

# 当前按下的按钮ID（实时状态）
uint32[] pressed_buttons  # 所有处于按下状态的按钮