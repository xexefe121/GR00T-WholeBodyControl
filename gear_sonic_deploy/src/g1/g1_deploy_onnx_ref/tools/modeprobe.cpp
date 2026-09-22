// Reports whether Unitree's motion-control service holds the joints.  Read-only.
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <iostream>
#include <string>
int main(int argc, char** argv) {
  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  unitree::robot::b2::MotionSwitcherClient msc; msc.SetTimeout(5.0f); msc.Init();
  std::string form, name; msc.CheckMode(form, name);
  std::cout << "mode name=\"" << name << "\"" << (name.empty() ? "  -> NO MODE HELD" : "  -> A MODE IS HELD") << "\n";
  return 0;
}
