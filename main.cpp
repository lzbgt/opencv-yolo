/**
 * Author: bruce.lu (lzbgt@icloud.com)
 * 
 * 
 * 
 */

#include <clipp.h>

#include <sstream>

#include "event_detection.hpp"
#include "yolo.hpp"

using namespace clipp;
using namespace std;

int main(int argc, char *argv[]) {
    float fConfident = 0.2f;
    bool bVerbose = false;
    bool help = false;
    bool bCont = false;
    int wrap = 10;
    string sInput, sOutput = "detect.jpg";
    string modelPath = ".";
    int gapSeconds = 4;

    auto cli = (option("-cl") & value("confidence level of detection, default: 0.2", fConfident),
                option("-vv", "--debug").set(bVerbose).doc("verbose prints"),
                option("-c", "--config-path") & value("model and configuration path", modelPath),
                option("-g", "--gap") & value("gap time for idle", gapSeconds),
                option("-h", "--help").set(help).doc("print this help info"),
                value("input path", sInput));

    if (!parse(argc, argv, cli) || help) {
        stringstream s;
        s << make_man_page(cli, argv[0]);
        spdlog::info(s.str());
        exit(0);
    }

    spdlog::set_level(spdlog::level::debug);

    spdlog::info("detector: {} {} {}", fConfident, wrap, bVerbose);

    YoloDectect detector(modelPath, gapSeconds, bVerbose, fConfident, bCont, wrap);
    detector.process(sInput);
}