import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String
from perception.focus_tools.focuser import Focuser

class ManualFocusNode(Node):
    def __init__(self):
        super().__init__('manual_focus_node')

        self.declare_parameter('i2c_bus', 9)
        # The manual focus value (0 to 1000)
        # E.g. "ros2 param set /manual_focus_node focus_value 300"
        self.declare_parameter('focus_value', 200)
        
        i2c_bus = self.get_parameter('i2c_bus').value
        initial_focus = self.get_parameter('focus_value').value
        
        self.focuser = Focuser(i2c_bus)

        self.focuser.set(Focuser.OPT_FOCUS, initial_focus)
        self.get_logger().info(f"ManualFocusNode started. i2c_bus={i2c_bus}, initial focus={initial_focus}")

        # Add a parameter callback so node react to param updates in real time
        self.add_on_set_parameters_callback(self.param_callback)

    def param_callback(self, params):
        """
        This callback is triggered whenever a parameter is updated via
        `ros2 param set /manual_focus_node <param> <value>`.
        """
        successful = True
        reason = ""

        for param in params:
            if param.name == 'focus_value':
                # Check if it's in valid range [0..1000]
                new_focus = param.value
                if new_focus < 0 or new_focus > 1000:
                    successful = False
                    reason = "focus_value must be between 0 and 1000"
                    break
                
                # Set the lens
                self.focuser.set(Focuser.OPT_FOCUS, new_focus)
                self.get_logger().info(f"Focus updated to {new_focus}")
        
        # Return whether the update was successful
        result = SetParametersResult()
        result.successful = successful
        result.reason = reason
        return result

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
