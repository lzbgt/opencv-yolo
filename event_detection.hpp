#include <spdlog/spdlog.h>

#include <exception>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <thread>
#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "utils/safe_queue.hpp"

using json = nlohmann::json;
class EventDetection {
    SafeQueue<int> q;
    bool isRunning = false;
    json clients;
    json lastMessage;

    std::mutex _mut;

   public:
    static EventDetection& getInstance() {
        static EventDetection inst;
        return inst;
    }
    void notify(int cnt) {
        q.push(cnt);
    }
    void run() {
        if (!isRunning) {
            _run();
        }
    }

    json getLastMessage() {
        json ret;
        _mut.lock();
        if (lastMessage.count("event") == 0) {
            lastMessage["event"] = "face_off";
            lastMessage["cnt"] = 0;
        }
        ret["event"] = lastMessage["event"];
        ret["cnt"] = lastMessage["cnt"];
        _mut.unlock();
        return ret;
    }

    void setLastMessage(json msg) {
        _mut.lock();
        lastMessage = msg;
        _mut.unlock();
    }

   protected:
    EventDetection() {}

   private:
    void _run() {
        isRunning = true;
        auto th = std::thread([this]() {
            std::cout << "init ctx: " << std::endl;
            zmq::context_t ctx(1);
            std::cout << "create socket ctx: " << std::endl;
            zmq::socket_t sock(ctx, zmq::socket_type::router);
            int opt = 3;
            try {
                sock.setsockopt(ZMQ_ROUTING_ID, "face", 5);  // routeId
                std::string addr = "tcp://0.0.0.0:3015";
                std::cout << "binding: " << addr << std::endl;
                sock.bind(addr.c_str());
            } catch (std::exception& e) {
                std::cout << "exception: " << e.what() << std::endl;
                exit(1);
            }

            auto recvTh = std::thread([this, &sock]() {
                while (true) {
                    std::vector<zmq::message_t> recv_msgs;
                    std::cout << "receiving msg:\n";

                    zmq::recv_result_t result;

                    try {
                        result = zmq::recv_multipart(sock, std::back_inserter(recv_msgs));

                        for (auto& s : recv_msgs) {
                            auto pos = &s - &recv_msgs[0];
                            std::cout << "pos: " << pos << " " << s.to_string() << s.size() << std::endl;
                        }

                        std::cout << std::endl;
                        auto data = this->getLastMessage();

                        if (recv_msgs.size() >= 2 && data["cnt"] > 0) {
                            if (strncmp(recv_msgs[1].to_string().data(), "hello", 5) == 0) {
                                std::cout << "will response" << std::endl;
                                std::string cl = recv_msgs[0].to_string();
                                auto body = this->getLastMessage().dump();
                                std::array<zmq::const_buffer, 2> v = {
                                    zmq::buffer(cl),  //targetId
                                    zmq::buffer(body)};
                                zmq::send_multipart(sock, v);
                                spdlog::info("replied hello message {}", body);

                                if (!clients.contains(cl)) {
                                    clients[cl] = std::chrono::system_clock::now().time_since_epoch().count();
                                }
                            }
                        }
                    } catch (std::exception& e) {
                        spdlog::info("exception: {}", e.what());
                        // exit(1);
                    }
                    
                }
            });
            recvTh.detach();

            // send thread
            while (true) {
                json data;
                auto cnt = this->q.take();
                data["cnt"] = cnt;
                auto v = this->getLastMessage();
                if (cnt > 0 && v["cnt"] == 0) {
                    data["event"] = "face_on";
                } else if (cnt == 0 && v["cnt"] > 0) {
                    data["event"] = "face_off";
                } else {
                    // skip
                    //spdlog::info("skip send: {}, {}", data.dump(), v.dump());
                    continue;
                }

                this->setLastMessage(data);
                auto str = data.dump();

                spdlog::info("send: {}", str);

                auto buf = zmq::buffer(str.data(), str.size());

                for (auto& [k, v] : clients.items()) {
                    std::array<zmq::const_buffer, 2> v = {
                        zmq::buffer(k),
                        buf};
                    zmq::send_multipart(sock, v);
                    spdlog::info("sent: {}", str);
                }
            }
        });

        th.detach();
    }
};