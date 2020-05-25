#include "main.h"

int main() {

    // Load plugin for IE
    /*************************************************************************/
    InferenceEngine::Core core;
    /*************************************************************************/

    // Read IR
    /*************************************************************************/
    InferenceEngine::CNNNetwork net = core.ReadNetwork(model::MODEL_XML, model::MODEL_BIN);
    /*************************************************************************/

    // Prepare input blob
    /*************************************************************************/
    InferenceEngine::InputsDataMap input_map = net.getInputsInfo();
    std::cout << "Network input\tPrecision\tLayout\tColor format" << std::endl;
    size_t net_chan = 0, net_height = 0, net_width = 0;
    for (std::map<std::string, InferenceEngine::InputInfo::Ptr>::iterator it = input_map.begin(); it != input_map.end(); ++it) {
        // Get blob dimensions
        InferenceEngine::SizeVector input_blob_sz = it->second->getTensorDesc().getDims();
        net_chan = input_blob_sz[1];
        net_height = input_blob_sz[2];
        net_width = input_blob_sz[3];

        // Set precision
        it->second->setPrecision(InferenceEngine::Precision::U8);

        // Dump blob info
        std::cout << it->first << "\t" << it->second->getPrecision().name() << "\t\t" << it->second->getLayout() << "\t"\
                  << it->second->getPreProcess().getColorFormat() << std::endl;
    }
    /*************************************************************************/

    // Prepare output blob
    /*************************************************************************/
    InferenceEngine::OutputsDataMap output_map = net.getOutputsInfo();
    std::cout << "Network output\tPrecision" << std::endl;
    for (std::map<std::string, InferenceEngine::DataPtr>::iterator it = output_map.begin(); it != output_map.end(); ++it) {
        // Dump blog info
        std::cout << it->first << "\t" << it->second->getPrecision().name() << std::endl;
    }
    /*************************************************************************/

    // Load model to target device
    /*************************************************************************/
    std::map<std::string, std::string> config = {{CONFIG_KEY(CPU_THROUGHPUT_STREAMS), "1"}};
    InferenceEngine::ExecutableNetwork exec_net = core.LoadNetwork(net, model::DEVICE, config);
    /*************************************************************************/

    // INFERENCE LOOP
    /*************************************************************************/
    cv::VideoCapture cap = cv::VideoCapture(model::INPUT_VID);
    cv::Mat frame_data, old_frame_data, net_data;
    InferenceEngine::InferRequest infer_req = exec_net.CreateInferRequest();
    Benchmark benchmark;
    bool rc = false;
    uint32_t frame_num = 0;
    uint16_t frame_width, frame_height;
    while(true) {
        // Capture frame
        /*************************************************************************/
        benchmark.start();
        rc = cap.read(frame_data);
        if (rc == false) break;
        if (frame_num == 0) {
            frame_width = frame_data.cols;
            frame_height = frame_data.rows;
        }
        benchmark.end("capture");
        /*************************************************************************/

        // Pre-process frame
        /*************************************************************************/
        benchmark.start();
        cv::resize(frame_data, net_data, cv::Size(net_width, net_height));
        benchmark.end("pre-process");
        /*************************************************************************/

        // Infer
        /*************************************************************************/
        benchmark.start();
        InferenceEngine::Blob::Ptr input_blob = infer_req.GetBlob(model::INPUT_BLOB);
        common::fillBlob(input_blob, net_data);
        infer_req.Infer();
        benchmark.end("inference");
        /*************************************************************************/

        // Filter bounding boxes
        /*************************************************************************/
        benchmark.start();
        InferenceEngine::Blob::Ptr output_blob = infer_req.GetBlob(model::OUTPUT_BLOB);
        std::vector<BdBox> bd_box_vec;
        common::getBdBox(bd_box_vec, output_blob, frame_width, frame_height);
        common::nonMaxSup(bd_box_vec);
        benchmark.end("filter");
        /*************************************************************************/

        // Assign ID
        /*************************************************************************/
        /*************************************************************************/

        // Update counting
        /*************************************************************************/
        /*************************************************************************/

        // Show/store results
        /*************************************************************************/
        benchmark.start();
        for (std::vector<BdBox>::iterator it = bd_box_vec.begin(); it != bd_box_vec.end(); it++) {
            cv::rectangle(frame_data, cv::Point2f(it->xmin, it->ymin), cv::Point2f(it->xmax, it->ymax),
                          cv::Scalar(232, 35, 244));
        }
        cv::imshow("Tracker", frame_data);
        cv::waitKey(9);
        benchmark.end("show");
        /*************************************************************************/

        frame_num++;
        old_frame_data = frame_data;
    }
    std::cout << "End of program..." << std::endl;
    /*************************************************************************/
    return(0);
}
