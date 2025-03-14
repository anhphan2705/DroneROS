import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from msgs.msg import FocusStatus
from perception.focus_tools.focuser import Focuser

class ManualFocusNode(Node):
    def __init__(self):
        super().__init__('manual_focus_node')

        self.declare_parameter('i2c_bus', 9)
        self.declare_parameter('focus_interval', 30.0)
        self.declare_parameter('focus_value', 200)
        
        i2c_bus = self.get_parameter('i2c_bus').value
        self.reapply_interval = self.get_parameter('focus_interval').value
        self.current_focus = self.get_parameter('focus_value').value
        
        self.get_logger().info(
            f"ManualFocusNode started. i2c_bus={i2c_bus}, initial focus={self.current_focus}, "
            f"reapply_interval={self.reapply_interval}"
        )
        
        self.focus_info_publisher_ = self.create_publisher(FocusStatus, '/camera/status/focus_info', 10)
        
        self.focuser = Focuser(i2c_bus)
        self.apply_focus(self.current_focus)
        self.timer = self.create_timer(self.reapply_interval, self.reapply_focus)

        self.add_on_set_parameters_callback(self.param_callback)

    def param_callback(self, params):
        """
        Called when parameters are updated, e.g.:
          ros2 param set /manual_focus_node focus_value 300
          ros2 param set /manual_focus_node focus_interval 5.0
        """
        successful = True
        reason = ""

        for param in params:
            if param.name == 'focus_value':
                new_focus = param.value
                if new_focus < 0 or new_focus > 1000:
                    successful = False
                    reason = "focus_value must be [0..1000]"
                    break
                self.current_focus = new_focus
                self.apply_focus(self.current_focus)
                self.get_logger().info(f"Focus updated immediately to {new_focus}")

            elif param.name == 'focus_interval':
                new_interval = float(param.value)
                if new_interval <= 0:
                    successful = False
                    reason = "focus_interval must be > 0"
                    break
                self.reapply_interval = new_interval
                self.timer.cancel()
                self.timer = self.create_timer(self.reapply_interval, self.reapply_focus)
                self.get_logger().info(f"Reapply interval updated to {new_interval} seconds")

        result = SetParametersResult()
        result.successful = successful
        result.reason = reason
        return result

    def apply_focus(self, value: int):
        """
        Helper to apply a focus value to the lens.
        """
        self.focuser.set(self.focuser.OPT_FOCUS, value)
        self.publish_focus_info(value, self.reapply_interval)

    def reapply_focus(self):
        """
        Timer callback: Re-apply the same focus setting to keep lens from drifting.
        """
        self.apply_focus(self.current_focus)
        self.get_logger().info(f"Re-applied focus = {self.current_focus}")

    def publish_focus_info(self, focus_value: int, interval: float):
        """
        Publishes a single-field message (FocusInfo) to /camera/status/focus_info.
        """
        msg = FocusStatus()
        msg.focus_value = focus_value
        msg.reapply_interval = interval
        self.focus_info_publisher_.publish(msg)
        self.get_logger().info(f"Published focus_info: focus_value={focus_value}")

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ManualFocusNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()