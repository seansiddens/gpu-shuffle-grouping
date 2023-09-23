#include <iostream>
#include <vector>

#include "easyvk.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION // This line must appear in one .cpp file
#include "stb_image_write.h"

const uint32_t seed = 0xcafebabe;

bool shuffleTest(easyvk::Device device, size_t numItems, bool printOutput) {
    // Load shader code.
    std::vector<uint32_t> spvCode = 
    #include "build/shuffle.cinit"
    ;
    auto entry_point = "shuffle";

    // Shuffle 16 things.
    uint32_t numBits = uint32_t(std::ceil(std::log2(float(numItems))));
    std::cout << "num index bits: " << numBits << '\n';
    auto indexBitsBuf = easyvk::Buffer(device, 1, sizeof(uint32_t));
    indexBitsBuf.store<uint32_t>(0, numBits);
    auto seedBuf = easyvk::Buffer(device, 1, sizeof(uint32_t));
    seedBuf.store<uint32_t>(0, seed);
    auto outputBuf = easyvk::Buffer(device, numItems, sizeof(uint32_t));
    outputBuf.clear();
    auto successFlagBuf = easyvk::Buffer(device, 1, sizeof(uint32_t));
    successFlagBuf.store<uint32_t>(0, 1); // set to true.

    // Init kernel program.
    std::vector<easyvk::Buffer> kernelInputs = {indexBitsBuf,
                                                seedBuf,
                                                outputBuf, 
                                                successFlagBuf};
    auto program = easyvk::Program(device, spvCode, kernelInputs);
    program.setWorkgroups(numItems);
    program.setWorkgroupSize(1); 
    program.initialize(entry_point);

    // Launch kernel.
    program.run();

    // Check and print results.
    if (successFlagBuf.load<uint32_t>(0) != 1) {
        // the shuffle failed.
        return false;
    }

    if (printOutput) {
        bool first = true;
        for (int i = 0; i < numItems; i++) {
            std::cout << (first ? "" : ", " ) << outputBuf.load<uint32_t>(i);
            first = false;
        }
        std::cout << "\n\n";
    } 

    return true;
}


int main() {
    size_t deviceIndex = 0;
    // Query device properties.
	auto instance = easyvk::Instance(true);
    auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
    auto deviceName = device.properties.deviceName;
    std::cout << "Using device: " << deviceName << "\n";

    // NOTE: log_2(numItems) must be even!
    if (!shuffleTest(device, 12, true)) {
        std::cerr << "Shuffle test failed!\n";
        return 1;
    }

    device.teardown();
    instance.teardown();
    return 0;
}