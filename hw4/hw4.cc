#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <pthread.h>
#include <time.h>
#include <thread>
#include <chrono>
#include <queue>
#include <unistd.h>

#define ascending 0
#define descending 1

using namespace std;

typedef std::pair<std::string, int> Item;
typedef std::pair<int, int> int_int;
typedef std::pair<int, string> record;
typedef std::pair<string, int> string_int;

string job_name;
int num_reducer;
unsigned int delay;
int chunk_size;
int total_chunk = 0;
string input_filename;
string output_dir;
string inter_dir = "./inter_file/";
string reduce_file_dir = "./reduce_file/";
string locality_config_filename;

int stop_mapping;
int stop_reducing;
int thread_available;
int ncpus;
int queue_size;
int inter_file_id = 1;
queue<pair<int, int>> chunk_waiting_queue;

queue<pair<int, int>> mapper_complete_info;
queue<pair<int, int>> reducer_complete_info;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex3 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_inter = PTHREAD_MUTEX_INITIALIZER;

int sorting = ascending;

struct cmp_{
    bool operator()(const string& k1, const string& k2) const{
        if(sorting == ascending){
            return k1 < k2;
        }
        else{
            return k1 > k2;
        }
    }
};

int cal_time(struct timespec start_time, struct timespec end_time)
{
    struct timespec temp;
    if ((end_time.tv_nsec - start_time.tv_nsec) < 0) {
        temp.tv_sec = end_time.tv_sec-start_time.tv_sec-1;
        temp.tv_nsec = 1000000000 + end_time.tv_nsec - start_time.tv_nsec;
    } else {
        temp.tv_sec = end_time.tv_sec - start_time.tv_sec;
        temp.tv_nsec = end_time.tv_nsec - start_time.tv_nsec;
    }
    double exe_time = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    return (int)(exe_time+0.5);
}

void* mapper(void* t){
    int* tid = (int*)t;
    int chunkID;
    int flag;
    pair<int, int> p;

    struct timespec start_time, end_time, temp;
    int chunk_time;

    while(1){
        pthread_mutex_lock(&mutex);
        if(queue_size > 0){
            p = chunk_waiting_queue.front();
            chunk_waiting_queue.pop();
            queue_size--;
            thread_available--;
            pthread_mutex_unlock(&mutex);

            clock_gettime(CLOCK_MONOTONIC, &start_time);

            if(p.second == 1) sleep(delay);

            chunkID = p.first;
            // cout << "chunkID: " << chunkID << endl;
            int start_pos = (chunkID-1) * chunk_size; // read text begin from #start_pos line
            int i = 0;
            std::ifstream input_file(input_filename);
            string input_line; 
            queue<record> records;
            map<string, int> word_count;

            // input split function
            while(i != start_pos){
                getline(input_file, input_line);
                i++;
            }
            for(int j = 0; j < chunk_size; j++){
                record r;
                getline(input_file, input_line);
                r.first = start_pos + j;
                r.second = input_line;
                records.push(r);
            }

            // map function
            for(int k = 0; k < chunk_size; k++){
                int pos = 0;
                record r = records.front();
                records.pop();
                string l = r.second;
                vector<string> words;
                string word;

                // while ((pos = l.find(" ")) != std::string::npos){
                //     word = l.substr(0, pos);
                //     words.push_back(word);

                //     l.erase(0, pos + 1);
                // }
                // if (!l.empty())
                //     words.push_back(l);

                // for (auto word : words){
                //     if (word_count.count(word) == 0){
                //         word_count[word] = 1;
                //     }
                //     else{
                //         word_count[word]++;
                //     }
                // }    
                while ((pos = l.find(" ")) != std::string::npos){
                    word = l.substr(0, pos);
                    if (word_count.count(word) == 0){
                        word_count[word] = 1;
                    }
                    else{
                        word_count[word]++;
                    }

                    l.erase(0, pos + 1);
                }

                if (word_count.count(l) == 0){
                    word_count[l] = 1;
                }
                else{
                    word_count[l]++;
                }
            }

            
            std::ofstream out(inter_dir + "inter" + "-" + std::to_string(chunkID) + ".out");
            for(Item item : word_count){
                out << item.first << " " << item.second << "\n";
            }

            pthread_mutex_lock(&mutex_inter);
            inter_file_id++;
            pthread_mutex_unlock(&mutex_inter);

            clock_gettime(CLOCK_MONOTONIC, &end_time);
            chunk_time = cal_time(start_time, end_time);

            pthread_mutex_lock(&mutex);
            mapper_complete_info.push(make_pair(chunkID, chunk_time));
            thread_available++;
            pthread_mutex_unlock(&mutex);
        }
        else{
            pthread_mutex_unlock(&mutex);
        }

        if(stop_mapping == 1 && inter_file_id == total_chunk + 1 && thread_available == ncpus - 1){
            
            break; 
        } 
    }
    pthread_exit(NULL);
}

int main(int argc, char **argv)
{
    int rc, rank, size;
	rc = MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);

    job_name = std::string(argv[1]);
    num_reducer = std::stoi(argv[2]);
    delay = std::stoul(argv[3]);
    input_filename = std::string(argv[4]);
    chunk_size = std::stoi(argv[5]);
    locality_config_filename = std::string(argv[6]);
    output_dir = std::string(argv[7]);

    std::map<std::string, int> word_count;
    // read words file
    std::string line;

    int num_task_tracker = size - 1;

    // job tracker
    if(rank == size - 1){
        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        std::ofstream log_out(output_dir + job_name + "-log.out");
        log_out << time(nullptr) << ",Start_Job," << job_name << "," << size << "," << ncpus << "," << num_reducer << "," << delay << "," << input_filename << "," << chunk_size << "," << locality_config_filename << "," << output_dir << endl;

        std::ifstream locality_file(locality_config_filename);
        string locality_line;
        std::vector<int_int> chunk_queue;

        int i = 1;
        while(getline(locality_file, locality_line)){
            int pos = 0;
            int line_len = locality_line.length();
            int chunk_id, chunk_pos;
            int_int c;

            pos = locality_line.find(" ");
            chunk_id = stoull(locality_line.substr(0, pos));
            chunk_pos = stoull(locality_line.substr(pos+1)) % num_task_tracker;
            
            c.first = chunk_id;
            c.second = chunk_pos;
            chunk_queue.push_back(c);
            total_chunk++;
        }

        // MPI communication
        // while there is still some chunk not being handled
        while(!chunk_queue.empty()){
            int from_node; // whose request
            int chunk_to_give[3]; // which chunk to give
            int index; // the position of the chunk in the queue
            int flag = 0;
            MPI_Status status;
            MPI_Recv(&from_node, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            for(int i = 0; i < chunk_queue.size(); i++){
                if(chunk_queue.at(i).second == from_node){
                    chunk_to_give[0] = chunk_queue.at(i).first;
                    chunk_to_give[1] = chunk_queue.at(i).second;
                    chunk_to_give[2] = from_node;
                    index = i;
                    flag = 1;
                    break;
                }
            }
            if(flag == 0){
                chunk_to_give[0] = chunk_queue.front().first;
                chunk_to_give[1] = chunk_queue.front().second;
                chunk_to_give[2] = from_node;
                index = 0;
                // sleep(delay);
            }

            log_out << time(nullptr) << ",Dispatch_MapTask," << chunk_to_give[0] << "," << from_node << endl;
            MPI_Send(&chunk_to_give, 3, MPI_INT, from_node, 0, MPI_COMM_WORLD);
            chunk_queue.erase(chunk_queue.begin() + index);
        }

        // the end
        stop_mapping = -1;
        for(int i = 0; i < size - 1; i++){
            MPI_Send(&stop_mapping, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        int count = 0;
        int chunk_info[2];
        while(count < total_chunk){
            MPI_Recv(&chunk_info, 2, MPI_INT, MPI_ANY_SOURCE, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            log_out << time(nullptr) << ",Complete_MapTask," << chunk_info[0] << "," << chunk_info[1] << "\n";
            count++;
        }

        int start_partition = 0;
        MPI_Recv(&start_partition, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /*
        **********************
          partition function
        **********************
        */

        struct timespec start_time_shuffle, end_time_shuffle;
        clock_gettime(CLOCK_MONOTONIC, &start_time_shuffle);
        log_out << time(nullptr) << ",Start_Shuffle," << to_string(total_chunk) << endl;

        vector<string_int> inter_pair_list[num_reducer];
        for(int c = 1; c <= total_chunk; c++){
            string path = inter_dir + "inter-" + std::to_string(c) + ".out";
            string inter_line;
            std::ifstream inter_file(path);

            while(getline(inter_file, inter_line)){
                int pos = 0;
                int inter_line_len = inter_line.length();
                string key;
                int value;
                string_int o;

                pos = inter_line.find(" ");
                key = inter_line.substr(0, pos);
                value = stoi(inter_line.substr(pos+1));
                o.first = key;
                o.second = value;
                int which_reducer = (key[0] - 'A') % num_reducer;
                inter_pair_list[which_reducer].push_back(o);
            }
        }

        for(int f = 0; f < num_reducer; f++){
            std::ofstream out(reduce_file_dir + "reduce" + "-" + std::to_string(f+1) + ".out");
            for(string_int item : inter_pair_list[f]){
                // if(item.first == "Aulacomnium"){
                //     cout << "Aulacomnium: " << item.second << endl;
                // }
                out << item.first << " " << item.second << "\n";
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &end_time_shuffle);
        int shuffle_time = cal_time(start_time_shuffle, end_time_shuffle);
        cout << "start: " << start_time.tv_sec << ", end: " << end_time.tv_sec << endl;
        log_out << time(nullptr) << ",Finish_Shuffle," << to_string(shuffle_time) << endl;

        // dispatch reduce task
        for(int r = 0; r < num_reducer; r++){
            int from_node;
            int task_num = r;
            MPI_Status status;
            MPI_Recv(&from_node, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &status);
            MPI_Send(&task_num, 1, MPI_INT, from_node, 7, MPI_COMM_WORLD);
            log_out << time(nullptr) << ",Dispatch_ReduceTask," << task_num+1 << "," << from_node << endl;
        }

        stop_reducing = -1;
        for(int t = 0; t < size-1; t++){
            int from;
            MPI_Recv(&from, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&stop_reducing, 1, MPI_INT, from, 7, MPI_COMM_WORLD);
        }

        int count_reduce = 0;
        int reduce_info[2];
        while(count_reduce < num_reducer){
            MPI_Recv(&reduce_info, 2, MPI_INT, MPI_ANY_SOURCE, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            log_out << time(nullptr) << ",Complete_ReduceTask," << reduce_info[0] << "," << reduce_info[1] << "\n";
            count_reduce++;
        }
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        int total_program_time = cal_time(start_time, end_time);
        log_out << time(nullptr) << ",Finish_Job," << to_string(total_program_time) << endl;
    }

    /*
    **********************
        task tracker
    **********************
    */
    else{
        int rc;
        int chunk_to_process[3]; // the chunk to process
        thread_available = ncpus - 1;
        stop_mapping = 0;
        queue_size = 0;

        pthread_t mapper_threads[ncpus - 1];
        pthread_t reducer_thread;
        int ID[ncpus - 1];

        for(int i = 0; i < ncpus - 1; i++){
            ID[i] = i;
            rc = pthread_create(&mapper_threads[i], NULL, mapper, (void*)&ID[i]);
        }

        /*
        **********************
                mapper
        **********************
        */

        while(1){
            pthread_mutex_lock(&mutex);
            int flag = 0;
            if(thread_available > 0){
                MPI_Send(&rank, 1, MPI_INT, size-1, 0, MPI_COMM_WORLD);
                MPI_Recv(&chunk_to_process, 3, MPI_INT, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(chunk_to_process[1] != chunk_to_process[2]){
                    flag = 1;
                };
                if(chunk_to_process[0] == -1){
                    stop_mapping = 1;
                    pthread_mutex_unlock(&mutex);
                }
                else{
                    chunk_waiting_queue.push(make_pair(chunk_to_process[0], flag));
                    queue_size++;
                    total_chunk++;
                    // cout << "get job by process: " << rank << ", chunk ID: " << chunk_to_process[0] << ", flag: " << flag << endl;
                    pthread_mutex_unlock(&mutex);
                } 
            }

            if(stop_mapping){
                for(int i = 0; i < ncpus - 1; i++){
                    pthread_join(mapper_threads[i], NULL);
                }

                while(!mapper_complete_info.empty()){
                    int chunk_info[2];
                    pair<int, int> info = mapper_complete_info.front();
                    mapper_complete_info.pop();
                    chunk_info[0] = info.first;
                    chunk_info[1] = info.second;
                    MPI_Send(&chunk_info, 2, MPI_INT, size-1, 5, MPI_COMM_WORLD);
                }
                if(rank == 0){
                    int start_partition = 1;
                    MPI_Send(&start_partition, 1, MPI_INT, size-1, 1, MPI_COMM_WORLD);
                }
                break;
            }
        }

        /*
        **********************
                reducer
        **********************
        */
        int local_reduce_tasks = 0;
        while(1){
            int task_num;
            MPI_Send(&rank, 1, MPI_INT, size-1, 3, MPI_COMM_WORLD);
            MPI_Recv(&task_num, 1, MPI_INT, size-1, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            struct timespec start_time_reduce, end_time_reduce;
            clock_gettime(CLOCK_MONOTONIC, &start_time_reduce);

            if(task_num == -1){
                break;
                // MPI_Finalize();
                // return 0;
            } 

            string reduce_path = reduce_file_dir + "reduce" + "-" + std::to_string(task_num+1) + ".out";
            std::ifstream reduce_file(reduce_path);
            vector<string_int> reduce_strings;
            string reduce_string;

            while(getline(reduce_file, reduce_string)){
                int pos = 0;
                int reduce_string_len = reduce_string.length();
                string key;
                int value;
                string_int o;

                pos = reduce_string.find(" ");
                key = reduce_string.substr(0, pos);
                value = stoi(reduce_string.substr(pos+1, reduce_string_len-pos-1));
                o.first = key;
                o.second = value;
                reduce_strings.push_back(o);
            }

            /*
            **********************
                sorting function
            **********************
            */
            if(sorting == ascending){
                std::sort(reduce_strings.begin(), reduce_strings.end(), [](const string_int &item1, const string_int &item2) -> bool
                {return item1.first < item2.first;});
            }
            else{
                std::sort(reduce_strings.begin(), reduce_strings.end(), [](const string_int &item1, const string_int &item2) -> bool
                {return item1.first > item2.first;});
            }
            
            // std::sort(reduce_strings.rbegin(), reduce_strings.rend());


            /*
            **********************
                group function
            **********************
            */
            vector<string_int>::iterator it;
            map<string, string, cmp_> group_result;
            for(it = reduce_strings.begin(); it != reduce_strings.end(); it++){
                string key = it.base()->first;
                int value = it.base()->second;

                if(group_result.count(key) == 0){
                    group_result[key] = to_string(value);
                }
                else{
                    group_result[key] += (" " + to_string(value));
                }
            }

            /*
            **********************
                reduce function
            **********************
            */
            map<string, int, cmp_> reduce_map;
            map<string, string, cmp_>::iterator iter = group_result.begin();
            while(iter != group_result.end()){
                int sum = 0;
                int pos = 0;
                string key = iter->first;
                string val_str = iter->second;
                string s;
                
                while ((pos = val_str.find(" ")) != std::string::npos){
                    s = val_str.substr(0, pos);
                    sum += stoi(s);
                    val_str.erase(0, pos + 1);
                }
                if(!val_str.empty()){
                    sum += stoi(val_str);
                }

                reduce_map[key] = sum;
                iter++;
            }

            clock_gettime(CLOCK_MONOTONIC, &end_time_reduce);
            int reduce_time = cal_time(start_time_reduce, end_time_reduce);
            reducer_complete_info.push(make_pair(task_num, reduce_time));
            local_reduce_tasks++;

            /*
            **********************
                output function
            **********************
            */

            std::ofstream out(output_dir + job_name + "-" + std::to_string(task_num+1) + ".out");
            map<string, int, cmp_>::iterator iter_output = reduce_map.begin();
            while(iter_output != reduce_map.end()){
                out << iter_output->first << " " << iter_output->second << '\n';
                iter_output++;
            }
        }
        int reduce_count = 0;
        while(reduce_count < local_reduce_tasks){
            int reduce_info[2];
            pair<int, int> p = reducer_complete_info.front();
            reducer_complete_info.pop();
            reduce_info[0] = p.first;
            reduce_info[1] = p.second;
            MPI_Send(&reduce_info, 2, MPI_INT, size-1, 6, MPI_COMM_WORLD);
            reduce_count++;
        }
    }
    MPI_Finalize();
    return 0;
}