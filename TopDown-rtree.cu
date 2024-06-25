#include<vector>
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h>
#include <sys/time.h>
// #include<algorithm>
#define M 10
#define MINERROR 0.00000001

using namespace std;


struct MBR{
    long long int x1,x2,y1,y2;
    // unsigned int x1,x2,y1,y2;
    unsigned long long int val;
    int start,end;
};

int cmp(const void *a,const void *b){
    return (*(MBR*)(a)).val > (*(MBR*)(b)).val;
}

int Overlap(MBR a,MBR b){
    // 无完全重叠判断
    // return  ((b.x1 <= a.x1 && a.x1 <= b.x2) && (b.x1 <= a.x2 && a.x2 <= b.x2)) * ((b.y1 <= a.y1 && a.y1 <= b.y2) && (b.y1 <= a.y2 && a.y2 <= b.y2));
    
    //完全重叠剪枝
    // return  (((b.x1 <= a.x1 && a.x1 <= b.x2) + (b.x1 <= a.x2 && a.x2 <= b.x2)) * ((b.y1 <= a.y1 && a.y1 <= b.y2) + (b.y1 <= a.y2 && a.y2 <= b.y2))) % 4;
    if(((unsigned long long)b.x2+a.x2-b.x1-a.x1)+MINERROR<(max(b.x2,a.x2) - min(b.x1,a.x1))){
		return 0;
	}
	if(((unsigned long long)b.y2+a.y2-b.y1-a.y1)+MINERROR<(max(b.y2,a.y2) - min(b.y1,a.y1))){
		return 0;
	}
	return 1;
}

__device__ int Overlap1(MBR a,MBR b){
    // 无完全重叠判断
    // return  ((b.x1 <= a.x1 && a.x1 <= b.x2) && (b.x1 <= a.x2 && a.x2 <= b.x2)) * ((b.y1 <= a.y1 && a.y1 <= b.y2) && (b.y1 <= a.y2 && a.y2 <= b.y2));
    
    //完全重叠剪枝
    // return  (((b.x1 <= a.x1 && a.x1 <= b.x2) + (b.x1 <= a.x2 && a.x2 <= b.x2)) * ((b.y1 <= a.y1 && a.y1 <= b.y2) + (b.y1 <= a.y2 && a.y2 <= b.y2))) % 4;
    if(((unsigned long long)b.x2+a.x2-b.x1-a.x1)+MINERROR<(max(b.x2,a.x2) - min(b.x1,a.x1))){
		return 0;
	}
	if(((unsigned long long)b.y2+a.y2-b.y1-a.y1)+MINERROR<(max(b.y2,a.y2) - min(b.y1,a.y1))){
		return 0;
	}
	return 1;
}

int RtreeSearch(int * nodes_index,MBR * nodes_mbr,int index,MBR target,int key,int * num,int & num1){
	int ans = -1;
    for(int i = 0;i < M;i++){
        num[0]++;
        if(nodes_index[index + i] != 0){
            // printf("%d %d (%d,%d  %d,%d) %d\n",index,nodes_index[index + i],nodes_mbr[index+i].x1,nodes_mbr[index+i].y1,nodes_mbr[index+i].x2,nodes_mbr[index+i].y2,Overlap(nodes_mbr[index + i],target));
            
            if(Overlap(nodes_mbr[index + i],target)){
                ans++;
                if(nodes_index[index + i] > 0){
                    ans += RtreeSearch(nodes_index,nodes_mbr,nodes_index[index + i],target,key,num,num1);
                }else{
                    num1++;
                    // printf("key = %d\n",key);
                }
            }

        }
    }
    return max(0,ans);
}

long long int RtreeSearch1(int * nodes_index,MBR * nodes_mbr,int index){
	long long int ans = 0;
    for(int i = 1;i < M && nodes_index[index + i] != 0;i++){
        for(int j = i - 1;j >= 0;j--){
            if(Overlap(nodes_mbr[index + i],nodes_mbr[index + j])){
                ans++;
            }
        }
    }
    for(int i = 0;i < M && nodes_index[index + i] != 0;i++){
        if(nodes_index[index + i] > 0){
            ans += RtreeSearch1(nodes_index,nodes_mbr,nodes_index[index + i]);
        }
    }
    return ans;
} 

__global__  void Construction_init(int * nodes_index,MBR * nodes_mbr,MBR * nums_g,int size,int offset){
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int threadSize = blockDim.x * gridDim.x;
    for(int i = tx;i < size;i += threadSize){
        nodes_index[i + offset] = -i - 1;
        nodes_mbr[i + offset].x1 = nums_g[i].x1;   
        nodes_mbr[i + offset].y1 = nums_g[i].y1;
        nodes_mbr[i + offset].x2 = nums_g[i].x2;   
        nodes_mbr[i + offset].y2 = nums_g[i].y2; 
    }
 
}

__global__  void Construction(int * nodes_index,MBR * nodes_mbr,int offset1,int offset2,int offset3){
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int threadSize = blockDim.x * gridDim.x;
    for(int i = offset1 + tx;i < offset2;i += threadSize){
        nodes_index[i] = 0;
        if((i - offset1) * M + offset2 < offset3 && nodes_index[(i - offset1) * M + offset2] != 0){
        	nodes_index[i] = (i - offset1) * M + offset2;
            nodes_mbr[i].x1 = nodes_mbr[(i - offset1) * M + offset2].x1;
            nodes_mbr[i].y1 = nodes_mbr[(i - offset1) * M + offset2].y1;
            nodes_mbr[i].x2 = nodes_mbr[(i - offset1) * M + offset2].x2;
            nodes_mbr[i].y2 = nodes_mbr[(i - offset1) * M + offset2].y2;
            for(int j = 1;j < M && (i - offset1) * M + offset2 + j < offset3 && nodes_index[(i - offset1) * M + offset2 + j] != 0;j++){
                nodes_mbr[i].x1 = min(nodes_mbr[i].x1,nodes_mbr[(i - offset1) * M + offset2 + j].x1);   
                nodes_mbr[i].y1 = min(nodes_mbr[i].y1,nodes_mbr[(i - offset1) * M + offset2 + j].y1);
                nodes_mbr[i].x2 = max(nodes_mbr[i].x2,nodes_mbr[(i - offset1) * M + offset2 + j].x2);   
                nodes_mbr[i].y2 = max(nodes_mbr[i].y2,nodes_mbr[(i - offset1) * M + offset2 + j].y2); 
            }
        }
    }
}
__global__  void Construction1(MBR * nums_g,int * nodes_index,MBR * nodes_mbr,int * offset,int p,int size){
    int tx = threadIdx.x;
    int threadSize = blockDim.x;
    for(int i = tx;i < size;i += threadSize){
        nodes_index[i + offset[p]] = -i - 1;
        nodes_mbr[i + offset[p]].x1 = nums_g[i].x1;   
        nodes_mbr[i + offset[p]].y1 = nums_g[i].y1;
        nodes_mbr[i + offset[p]].x2 = nums_g[i].x2;   
        nodes_mbr[i + offset[p]].y2 = nums_g[i].y2; 
    }
    __syncthreads();
    p--;
    while(p >= 0){
        for(int i = offset[p] + tx;i < offset[p + 1];i += threadSize){
            nodes_index[i] = 0;
            if((i - offset[p]) * M + offset[p + 1] < offset[p + 2] && nodes_index[(i - offset[p]) * M + offset[p + 1]] != 0){
                nodes_index[i] = (i - offset[p]) * M + offset[p + 1];
                nodes_mbr[i].x1 = nodes_mbr[(i - offset[p]) * M + offset[p + 1]].x1;
                nodes_mbr[i].y1 = nodes_mbr[(i - offset[p]) * M + offset[p + 1]].y1;
                nodes_mbr[i].x2 = nodes_mbr[(i - offset[p]) * M + offset[p + 1]].x2;
                nodes_mbr[i].y2 = nodes_mbr[(i - offset[p]) * M + offset[p + 1]].y2;
                for(int j = 1;j < M && (i - offset[p]) * M + offset[p + 1] + j < offset[p + 2] && nodes_index[(i - offset[p]) * M + offset[p + 1] + j] != 0;j++){
                    nodes_mbr[i].x1 = min(nodes_mbr[i].x1,nodes_mbr[(i - offset[p]) * M + offset[p + 1] + j].x1);   
                    nodes_mbr[i].y1 = min(nodes_mbr[i].y1,nodes_mbr[(i - offset[p]) * M + offset[p + 1] + j].y1);
                    nodes_mbr[i].x2 = max(nodes_mbr[i].x2,nodes_mbr[(i - offset[p]) * M + offset[p + 1] + j].x2);   
                    nodes_mbr[i].y2 = max(nodes_mbr[i].y2,nodes_mbr[(i - offset[p]) * M + offset[p + 1] + j].y2); 
                }
            }
        }
        p--;
        __syncthreads();
    }
}

__global__ void fun(MBR * g_nums,unsigned int ** g_arrays,unsigned int * g_index,int numsSize,int g_arraysSize,int n){
    
    int inx = blockIdx.x * blockDim.x + threadIdx.x;
    if(inx < g_arraysSize){
		for(int j = 0;j < 16;j++){
			g_arrays[inx][j] = 0;
			// g_arrays[j][inx] = 0;
		}
		int size = min(numsSize,(inx + 1) * (numsSize / g_arraysSize + (numsSize % g_arraysSize != 0)));
        for(int i = inx * (numsSize / g_arraysSize + (numsSize % g_arraysSize != 0));i < size;i++){
            g_index[i] = g_arrays[inx][(g_nums[i].val >> n) % 16]++;
			// g_index[i] = g_arrays[(g_nums[i] >> n) % 16][inx]++;
        }
		// for(int i = inx;i < numsSize; i += g_arraysSize){
			// g_index[i] = g_arrays[inx][(g_nums[i] >> n) % 16]++;
		// }
    }
}

__global__ void prefixBlock(unsigned int ** g_arrays,unsigned int ** g_arrays_change,int g_arraysSize,int k){
    int inx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadSize = blockDim.x * gridDim.x;
    int arr_threadSize = max(1,threadSize / g_arraysSize);
 
    for(int i = inx / g_arraysSize;i < 16;i += arr_threadSize) {
        
        for(int j = inx % g_arraysSize + k;j < g_arraysSize;j += threadSize) {	

                g_arrays_change[j][i] = g_arrays[j][i] + g_arrays[j - k][i];  
				// g_arrays_change[i][j] = g_arrays[i][j] + g_arrays[i][j - k];      
        }
    }
	// for(int i = inx + k;i < g_arraysSize;i += threadSize){
	// 	for(int j = 0;j < 16;j++)
	// 		g_arrays_change[i][j] = g_arrays[i][j] + g_arrays[i - k][j];
	// }
}

__global__ void arraysChangeBlock(unsigned int ** g_arrays,unsigned int ** g_arrays_change,int g_arraysSize,int k){
	int inx = blockIdx.x * blockDim.x + threadIdx.x;
	int threadSize = blockDim.x * gridDim.x;
	for(int i = inx + k;i < g_arraysSize; i += threadSize){
		for(int j = 0;j < 16;j++){
			g_arrays[i][j] = g_arrays_change[i][j];
		}
	}
	// int arr_threadSize = threadSize / g_arraysSize;
	// for(int i = inx / g_arraysSize;i < 16;i += arr_threadSize){
	// 	// int i = inx / g_arraysSize;
	// 	for(int j = inx % g_arraysSize + k;j < g_arraysSize;j += threadSize){
	// 		g_arrays[i][j] = g_arrays_change[i][j];
	// 	}
	// }
}

__global__ void prefixGrid(unsigned int * g_arrays,unsigned int * g_arrays_change,int k){
	int inx = blockIdx.x * blockDim.x + threadIdx.x;
	int threadSize = blockDim.x * gridDim.x;
	for(int i = inx + k;i < 16;i += threadSize){
		g_arrays_change[i] = g_arrays[i] + g_arrays[i - k];
	}
}

__global__ void arraysChangeGrid(unsigned int * g_arrays,unsigned int * g_arrays_change,int k){
	int inx = blockIdx.x * blockDim.x + threadIdx.x;
	int threadSize = gridDim.x * blockDim.x;
	for(int i = inx + k;i < 16;i += threadSize){
		g_arrays[i] = g_arrays_change[i];
	}
}

__global__ void prefixGrid1(unsigned int ** g_arrays,unsigned int ** g_arrays_change,int index,int k){
	int inx = blockIdx.x * blockDim.x + threadIdx.x;
	int threadSize = blockDim.x * gridDim.x;
	for(int i = inx + k;i < 16;i += threadSize){
		g_arrays_change[i][index] = g_arrays[i][index] + g_arrays[i - k][index];
	}
}	

__global__ void arraysChangeGrid1(unsigned int** g_arrays,unsigned int ** g_arrays_change,int index,int k){
	int inx = blockIdx.x * blockDim.x + threadIdx.x;
	int threadSize = blockDim.x * gridDim.x;
	for(int i = inx + k;i < 16;i += threadSize){
		g_arrays[i][index] = g_arrays_change[i][index];
	}
}

__global__ void sort(MBR * g_nums,MBR * g_nums_change,unsigned int * g_index,unsigned int ** g_arrays,int size,int g_arraysSize,int n){
	int inx = blockIdx.x * blockDim.x + threadIdx.x;
	int threadSize = gridDim.x * blockDim.x;
	int index;
	int num = size / g_arraysSize + (size % g_arraysSize != 0);
	for(int i = inx; i < size;i += threadSize){
		index = i / num == 0 ? 0 : g_arrays[i / num - 1][(g_nums[i].val >> n) % 16];
		index += (g_nums[i].val >> n) % 16 ? g_arrays[g_arraysSize - 1][(g_nums[i].val >> n) % 16 - 1] : 0;
		// index = i / num == 0 ? 0 : g_arrays[(g_nums[i] >> n) % 16][i / num - 1];
		// index += (g_nums[i] >> n) % 16 ? g_arrays[(g_nums[i] >> n) % 16 - 1][g_arraysSize - 1] : 0;
		g_nums_change[index + g_index[i]] = g_nums[i];
	}
}

__global__ void sortX(MBR * nums_g,MBR * nums_g_temp,int size,int numsSize){
    int inx = threadIdx.x + blockDim.x * blockIdx.x;
    int threadSize = blockDim.x * gridDim.x;
    int begin,end;
    int p;
    for(int i = inx;i < numsSize;i += threadSize){
        begin = i / size * size;
        end = i / size * size + size;
        p = 0;
        while(begin < end){
            if(nums_g[i].x1 > nums_g[begin].x1 || (nums_g[i].x1 == nums_g[begin].x1 && i > begin)){
                p++;
            }
            begin++;
        }
        nums_g_temp[i / size * size + p] = nums_g[i];
    }
}

__global__ void sortY(MBR * nums_g,MBR * nums_g_temp,int size,int numsSize){
    int inx = threadIdx.x + blockDim.x * blockIdx.x;
    int threadSize = blockDim.x * gridDim.x;
    int begin,end;
    int p;
    for(int i = inx;i < numsSize;i += threadSize){
        begin = i / size * size;
        end = i / size * size + size;
        p = 0;
        while(begin < end){
            if(nums_g[i].y1 > nums_g[begin].y1 || (nums_g[i].y1 == nums_g[begin].y1 && i > begin)){
                p++;
            }
            begin++;
        }
        nums_g_temp[i / size * size + p] = nums_g[i];
    }
}

__global__ void change(MBR * nums_g,MBR * nums_g_temp,int size){
    int inx = threadIdx.x + blockDim.x * blockIdx.x;
    int threadSize = blockDim.x * gridDim.x;
    for(int i = inx;i < size;i += threadSize){
        nums_g[i] = nums_g_temp[i];
    }
}

__global__ void fun(MBR * nums_g,int * nodes_index,MBR * nodes_mbr,int size,int numsSize,int offset,int nodeSize){
    int inx = threadIdx.x + blockDim.x * blockIdx.x;
    int threadSize = blockDim.x * gridDim.x;
    int begin,end;
    for(int i = inx;i < nodeSize;i += threadSize){
        // nodes_index[i + offset] = offset + numsSize / size + i * M;
        begin = i * size;
        end = min(numsSize,i * size + size);
        if(begin < numsSize){
            nodes_index[i + offset] = offset + nodeSize + i * M;
            nodes_mbr[i + offset].x1 = nums_g[begin].x1;
            nodes_mbr[i + offset].x2 = nums_g[begin].x2;
            nodes_mbr[i + offset].y1 = nums_g[begin].y1;
            nodes_mbr[i + offset].y2 = nums_g[begin].y2;
        }else{
            nodes_index[i + offset]  = 0;
        }
        nodes_mbr[i + offset].start = begin;
        nodes_mbr[i + offset].end = end - 1;
        begin++;
        while(begin < end){
            nodes_mbr[i + offset].x1 = min(nodes_mbr[i + offset].x1,nums_g[begin].x1);
            nodes_mbr[i + offset].x2 = max(nodes_mbr[i + offset].x2,nums_g[begin].x2);
            nodes_mbr[i + offset].y1 = min(nodes_mbr[i + offset].y1,nums_g[begin].y1);
            nodes_mbr[i + offset].y2 = max(nodes_mbr[i + offset].y2,nums_g[begin].y2);
            begin++;
        }
    }
}

__global__ void fun1(int * nodes_index,int numsSize,int offset){
    int inx = threadIdx.x + blockDim.x * blockIdx.x;
    int threadSize = blockDim.x * gridDim.x;
    for(int i = inx;i < numsSize;i += threadSize){
        nodes_index[i + offset] = -i - 1;
    }
}

__global__ void fun2(int * nodes_index,int start,int end){
     int inx = threadIdx.x + blockDim.x * blockIdx.x;
    int threadSize = blockDim.x * gridDim.x;
    for(int i = start + inx;i < end;i += threadSize){
        nodes_index[i] = 0;
    }
}

__global__ void query1(int * nodes_index,MBR * nodes_mbr,int * query_qeue,int * query_qeue_len,MBR * targets,int querysize,int queryarrsize){
    int inx = threadIdx.x;
    int bix = blockIdx.x;
    int threadSize = blockDim.x;
    int blockSize = gridDim.x;
    int size;
    MBR target;
    for(int i = bix;i < querysize;i+= blockSize){
        size = query_qeue_len[i];
        __syncthreads();
        if(inx == 0)
            query_qeue_len[i] = 0;
        __syncthreads();
        target = targets[i];
        for(int j = inx;j < size;j += threadSize){
            for(int k = 0;k < M;k++){
                if(query_qeue[i * queryarrsize + j] + k != 0 && Overlap1(target,nodes_mbr[query_qeue[i * queryarrsize + j] + k]) && nodes_index[query_qeue[i * queryarrsize + j] + k] > 0){
                    query_qeue[i * queryarrsize + atomicAdd(&query_qeue_len[i],1)] = nodes_index[query_qeue[i * queryarrsize + j] + k];
                }
            }
        }
    }
}

__global__ void check(bool * flag_g,int * query_qeue_len,int querysize){
    flag_g[0] = false;
    for(int i = 0;i < querysize;i++)
        if(query_qeue_len[i] > 0)
            flag_g[i] = true;
}

__global__ void query(int * nodes_index,MBR * nodes_mbr,int * query_qeue,int * query_qeue_len,MBR * targets,int * prfix,int querysize,int queryarrsize){
    int inx = threadIdx.x;
    int bix = blockIdx.x;
    int threadSize = blockDim.x;
    int blockSize = gridDim.x;
    int size;
    MBR target;
    for(int i = bix;i < querysize;i+= blockSize){
        size = query_qeue_len[i];
        target = targets[i];
        for(int j = inx;j < size;j += threadSize){
            for(int k = 0;k < M;k++){
                prfix[i * queryarrsize + j * M + k] = (query_qeue[i * queryarrsize + j] + k != 0 && Overlap1(target,nodes_mbr[query_qeue[i * queryarrsize + j] + k]) && nodes_index[query_qeue[i * queryarrsize + j] + k] > 0);      
            }
        }
    }
}

__global__ void getprifixsum(int * query_qeue_len,int * prfix,int querysize,int queryarrsize){
    int inx = threadIdx.x;
    int bix = blockIdx.x;
    int threadSize = blockDim.x;
    int blockSize = gridDim.x;
    int size;
    extern __shared__ int temp[];
    for(int i = bix;i < querysize;i += blockSize){
        size = query_qeue_len[i] * 4;
        for(int j = 1;j < size;j <<= 1){
            for(int k = inx + j;k < size;k += threadSize){
                temp[k] = prfix[i * queryarrsize + k] + prfix[i * queryarrsize + k - j];
            }
            __syncthreads();
            for(int k = inx + j;k < size;k += threadSize){
                prfix[i * queryarrsize + k] = temp[k];
            }
            __syncthreads();
        }
    }
}

__global__ void getnewqueue(int * nodes_index,int * query_qeue,int * query_qeue_len,int * prfix,int querysize,int queryarrsize){
    int inx = threadIdx.x;
    int bix = blockIdx.x;
    int threadSize = blockDim.x;
    int blockSize = gridDim.x;
    int size;
    extern __shared__ int temp[];
    for(int i = bix;i < querysize;i += blockSize){
        size = query_qeue_len[i] * 4;
        for(int j = inx + 1;j < size;j += threadSize){
            if(prfix[i * queryarrsize + j] != prfix[i * queryarrsize + j - 1]){
                  temp[prfix[i * queryarrsize + j - 1]] = nodes_index[query_qeue[i * queryarrsize + j / 4]] + (j % 4);          
            }
        }
        __syncthreads();
        if(inx == 0)
            query_qeue_len[i] = prfix[i * queryarrsize + size - 1];
        size = prfix[i * queryarrsize + size - 1];
        for(int j = 0;j < size;j += threadSize){
            query_qeue[i * queryarrsize + j] = temp[j];
        }
    }
}

__global__ void init(MBR * nums,int numsSize,int offset,int * query_qeue,int *query_qeue_len,MBR * targets,int querysize,int queryarrsize){
    int inx = threadIdx.x + blockDim.x * blockIdx.x;
    int threadSize = blockDim.x * gridDim.x;
    for(int i = inx;i < querysize;i += threadSize){
        query_qeue[i * queryarrsize] = 0;
        if(i + offset < numsSize){
            query_qeue_len[i] = 1;
            targets[i] = nums[i + offset];
        }else
            query_qeue_len[i] = 0;
        // if(inx < 1)
        //     printf("%d %d %d %d\n",nums[i + offset].x1,nums[i + offset].y1,nums[i + offset].x2,nums[i + offset].y2);
    }
}

__global__ void changeY(MBR * nums,int size,int sortarrsize){
    int inx = threadIdx.x + blockDim.x * blockIdx.x;
    int threadSize = blockDim.x * gridDim.x;
    for(int i = inx;i < size;i += threadSize){
        // nums[i].val = nums[i].y1 + (i / sortarrsize) * ((unsigned long long int)(1) << 32);
        nums[i].val = nums[i].y1 + (unsigned long long int)(i / sortarrsize) * 1e12;
    }
}

__global__ void changeX(MBR * nums,int size,int sortarrsize){
    int inx = threadIdx.x + blockDim.x * blockIdx.x;
    int threadSize = blockDim.x * gridDim.x;
    for(int i = inx;i < size;i += threadSize){
        // nums[i].val = nums[i].x1 + (i / sortarrsize) * ((unsigned long long int)(1) << 32);
        nums[i].val = nums[i].x1 + (unsigned long long int)(i / sortarrsize) * 1e12;
    }
}

int main(){
    int size = 1e3;
    FILE *fp;
    fp=fopen("data/Rect1024.txt","r"); 
    fscanf(fp, "%d", &size);
    // size = min(size,100000);
    // size = 1024 * 1024 / 4;
    vector<int> offset;
    MBR * nums = (MBR*)malloc(sizeof(MBR) * size);
    srand(5);
    int key;
    for(int i = 0;i < size;i++){
        fscanf(fp, "%lld %lld", &nums[i].x1,&nums[i].x2);
    	fscanf(fp, "%lld %lld", &nums[i].y1,&nums[i].y2);
        fscanf(fp, "%d", &key);
        // if(nums[i].x2 < nums[i].x1 || nums[i].y2 < nums[i].y1){
            // if(i < 10)
            // printf("%lld %lld %lld %lld %lld %d\n",i,nums[i].x1,nums[i].x2,nums[i].y1,nums[i].y2,key);
            // return 0;
        // }
    }
    int b = 1;
    int num = size;
    while(num > 1){
        num = (num / M + (num % M != 0)) * M;
		offset.push_back(num);
        num /= M;
        b *= M;
        
    }
    offset.push_back(0);
    int l = 0,r = offset.size() - 1;
    while(l < r){
        offset[l] ^= offset[r];
        offset[r] ^= offset[l];
        offset[l] ^= offset[r];
        l++;
        r--;
    }
    for(int i = 1;i < offset.size();i++){
        offset[i] += offset[i - 1];
    }
    int * nodes_index;
    MBR * nodes_mbr;
    MBR * nums_g;
    MBR * nodes_mbr_c = (MBR*)malloc(sizeof(MBR) * offset.back());
    int* nodes_index_c = (int*)malloc(sizeof(int) * offset.back());
    // MBR * nums_g_temp;
    // qsort(nums,size,sizeof(MBR),cmp);
    cudaMalloc((void**)&nodes_index,sizeof(int) * offset.back());
    cudaMalloc((void**)&nodes_mbr,sizeof(MBR) * offset.back());
    // cudaMalloc((void**)&nums_g,sizeof(MBR) * size);
    // cudaMalloc((void**)&nums_g_temp,sizeof(MBR) * size);
    // cudaMemcpy(nums_g,nums,sizeof(MBR) * size,cudaMemcpyHostToDevice);

    struct timeval start, end;
    long elapsed;
    int p = 0;
    int a = 1;
    a = b;
    

    int arrSize = min(size,512 * 8);
    arrSize = 4000;
    int BLOCK = 500;
    MBR * g_nums[2] = {NULL};
	unsigned int ** arrays = (unsigned int **) malloc(sizeof(unsigned int *) * arrSize);
	unsigned int ** arrays_change = (unsigned int **)malloc(sizeof(unsigned int *) * arrSize);
	// unsigned int ** arrays = (unsigned int **)malloc(sizeof(unsigned int *) * 16);
	// unsigned int ** arrays_change = (unsigned int **)malloc(sizeof(unsigned int *) * 16);
	unsigned int ** g_arrays_change = NULL;
	unsigned int ** g_arrays = NULL;
	unsigned int * g_index = NULL;
    
    (cudaMalloc((void **)&g_nums[0],sizeof(MBR) * size));
	(cudaMalloc((void **)&g_nums[1],sizeof(MBR) * size));
    for(int i = 0;i < arrSize;i++){
		(cudaMalloc((void **)&arrays[i],sizeof(unsigned int) * 16));
		(cudaMalloc((void **)&arrays_change[i],sizeof(unsigned int) * 16));
	}
    (cudaMalloc((void ***)&g_arrays,sizeof(unsigned int *) * arrSize));
	(cudaMalloc((void ***)&g_arrays_change,sizeof(unsigned *) * arrSize));
	(cudaMalloc((void **)&g_index,sizeof(unsigned int) * size));

    cudaMemcpy(g_nums[0],nums,sizeof(MBR) * size,cudaMemcpyHostToDevice);
	cudaMemcpy(g_arrays,arrays,sizeof(unsigned int *) * arrSize,cudaMemcpyHostToDevice);
	cudaMemcpy(g_arrays_change,arrays_change,sizeof(unsigned int *) * arrSize,cudaMemcpyHostToDevice);

    
    gettimeofday(&start, NULL);
    

     while(p < offset.size() - 1){
        
        if(p & 1){
            for(int i = 0;i < size;i++){
                nums[i].val = nums[i].y1 + i / a * ((unsigned long long int)(1) << 32);
            }
        }
        else{
             for(int i = 0;i < size;i++){
                nums[i].val = nums[i].x1 + i / a * ((unsigned long long int)(1) << 32);
            }
        }
            
        qsort(nums,size,sizeof(MBR),cmp);
        a /= M;
        // 填写子节点信息
        int begin,end;
        for(int i = 0;i < offset[p + 1] - offset[p];i++){
            begin = i * a;
            end = min(size,i * a + a);
            if(begin < size){
                nodes_index_c[i + offset[p]] = offset[p + 1] + i * M;
                nodes_mbr_c[i + offset[p]].x1 = nums[begin].x1;
                nodes_mbr_c[i + offset[p]].x2 = nums[begin].x2;
                nodes_mbr_c[i + offset[p]].y1 = nums[begin].y1;
                nodes_mbr_c[i + offset[p]].y2 = nums[begin].y2;
            }
            else{
                nodes_index_c[i + offset[p]] = 0;
            }
            begin++;
            while(begin < end){
                nodes_mbr_c[i + offset[p]].x1 = min(nodes_mbr_c[i + offset[p]].x1,nums[begin].x1);
                nodes_mbr_c[i + offset[p]].x2 = max(nodes_mbr_c[i + offset[p]].x2,nums[begin].x2);
                nodes_mbr_c[i + offset[p]].y1 = min(nodes_mbr_c[i + offset[p]].y1,nums[begin].y1);
                nodes_mbr_c[i + offset[p]].y2 = max(nodes_mbr_c[i + offset[p]].y2,nums[begin].y2);
                begin++;
            }
        }
        p++;
    }
    for(int i = 0;i < size;i++){
        nodes_index_c[i + offset[offset.size() - 2]] = -i - 1;
        nodes_mbr_c[i + offset[offset.size() - 2]].x1 = nums[i].x1;
        nodes_mbr_c[i + offset[offset.size() - 2]].x2 = nums[i].x2;
        nodes_mbr_c[i + offset[offset.size() - 2]].y1 = nums[i].y1;
        nodes_mbr_c[i + offset[offset.size() - 2]].y2 = nums[i].y2;
    }
    for(int i = size + offset[offset.size() - 2];i < offset.back();i++)
        nodes_index_c[i] = 0;
    gettimeofday(&end, NULL);
    elapsed = (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec;
    printf("CPU Construction: %0.2lfms\n",(double)elapsed / 1000);
    // return 0;
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);
    
    p = 0;
    // a = 1;
    a = b;
    nums_g = g_nums[0];
    while(p < offset.size() - 1){
        // if(p & 1)
        //     sortY<<<8,512>>>(nums_g,nums_g_temp,size / a,size);
        // else
        //     sortX<<<8,512>>>(nums_g,nums_g_temp,size / a,size);
        if(p & 1)
            changeY<<<8,512>>>(nums_g,size,a);
        else
            changeX<<<8,512>>>(nums_g,size,a);

        for(int i = 0; i < 64; i = i + 4){
            // 每次循环从低到高取4位
            // 将512/1024个数据为一组，将数据分为若干组
            // 每组由一个线程组进行排序
            // 在每个线程组内计算本组数据的直方图
            // 合并各个直方图
            // 按照4位来移动数据排序
            fun<<<arrSize / BLOCK + (arrSize % BLOCK != 0),BLOCK>>>(g_nums[(i / 4) % 2],g_arrays,g_index,size,arrSize,i);

            for(int k = 1;k < arrSize; k <<= 1){
                prefixBlock<<<16 * arrSize / BLOCK,BLOCK>>>(g_arrays,g_arrays_change,arrSize,k);
                arraysChangeBlock<<<16 * arrSize / BLOCK,BLOCK>>>(g_arrays,g_arrays_change,arrSize,k);
            }
            for(int k = 1;k < 16;k <<= 1){
                prefixGrid<<<1,16>>>(arrays[arrSize - 1],arrays_change[arrSize - 1],k);
                arraysChangeGrid<<<1,16>>>(arrays[arrSize - 1],arrays_change[arrSize - 1],k);
                prefixGrid1<<<1,16>>>(g_arrays,g_arrays_change,arrSize - 1,k);
                arraysChangeGrid1<<<1,16>>>(g_arrays,g_arrays_change,arrSize - 1,k);
            }

            sort<<<512,512>>>(g_nums[(i / 4) % 2],g_nums[(i / 4  + 1) % 2],g_index,g_arrays,size,arrSize,i);
        }

        
        // change<<<8,512>>>(nums_g,nums_g_temp,size);
        a /= M;
        // 填写子节点信息
        fun<<<8,512>>>(nums_g,nodes_index,nodes_mbr,a,size,offset[p],offset[p + 1] - offset[p]);
        p++;
    }
    fun1<<<8,512>>>(nodes_index,size,offset[offset.size() - 2]);
    fun2<<<8,512>>>(nodes_index,offset[offset.size() - 2] + size,offset.back());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    
    cudaMemcpy(nums,nums_g,sizeof(MBR) * size,cudaMemcpyDeviceToHost);

    float time_kernel; 
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    printf("GPU Construction time %0.2fms\n", time_kernel);
    
    cudaMemcpy(nodes_index_c,nodes_index,sizeof(int) * offset.back(),cudaMemcpyDeviceToHost);
    cudaMemcpy(nodes_mbr_c,nodes_mbr,sizeof(MBR) * offset.back(),cudaMemcpyDeviceToHost);
    // cudaFree(nums_g_temp);
    // cudaFree(nums_g);

    int queryarrsize = 1000;
    int querysize = 5000;
    int * query_qeue;
    int * query_qeue_len;
    int * prfix;
    MBR * targets;
    cudaMalloc((void**)& query_qeue,sizeof(int) * querysize * queryarrsize);
    cudaMalloc((void**)& prfix,sizeof(int) * querysize * queryarrsize);
    cudaMalloc((void**)& query_qeue_len,sizeof(int) * querysize);
    cudaMalloc((void**)& targets,sizeof(MBR) * querysize);
    bool * flag_g;
    cudaMalloc((void**)& flag_g,sizeof(bool));
    bool flag = true;
    int len;
    float time_kernel1 = 0,time_kernel2 = 0,time_kernel3 = 0,time_kernel4; 
    cudaEvent_t start_event1, stop_event1;
    cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);
    for(int i = 0;i < size;i += querysize){
        init<<<128,512>>>(nums_g,size,i,query_qeue,query_qeue_len,targets,querysize,queryarrsize);
        flag = true;
        len = 0;
        while(len < offset.size() - 1){
            // printf("%d \n",len);
            // query1<<<8,512>>>(nodes_index,nodes_mbr,query_qeue,query_qeue_len,targets,querysize,queryarrsize);
            // check<<<1,1>>>(flag_g,query_qeue_len,querysize);
            // cudaMemcpy(&flag,flag_g,sizeof(bool),cudaMemcpyDeviceToHost);
            cudaEventCreate(&start_event1) ;
            cudaEventCreate(&stop_event1) ;
            cudaEventRecord(start_event1, 0);
            // query<<<128,512>>>(nodes_index,nodes_mbr,query_qeue,query_qeue_len,targets,prfix,querysize,queryarrsize);
            query1<<<128,512>>>(nodes_index,nodes_mbr,query_qeue,query_qeue_len,targets,querysize,queryarrsize);
            cudaEventRecord(stop_event1, 0);
            cudaEventSynchronize(stop_event1);
            cudaEventElapsedTime(&time_kernel4, start_event1, stop_event1);
            time_kernel1 += time_kernel4;

            // cudaEventCreate(&start_event1) ;
            // cudaEventCreate(&stop_event1) ;
            // cudaEventRecord(start_event1, 0);
            // getprifixsum<<<128,512,sizeof(int) * queryarrsize>>>(query_qeue_len,prfix,querysize,queryarrsize);
            // cudaEventRecord(stop_event1, 0);
            // cudaEventSynchronize(stop_event1);
            // cudaEventElapsedTime(&time_kernel4, start_event1, stop_event1);
            // time_kernel2 += time_kernel4;

            // cudaEventCreate(&start_event1) ;
            // cudaEventCreate(&stop_event1) ;
            // cudaEventRecord(start_event1, 0);
            // getnewqueue<<<128,512,sizeof(int) * queryarrsize>>>(nodes_index,query_qeue,query_qeue_len,prfix,querysize,queryarrsize);
            // cudaEventRecord(stop_event1, 0);
            // cudaEventSynchronize(stop_event1);
            // cudaEventElapsedTime(&time_kernel4, start_event1, stop_event1);
            // time_kernel3 += time_kernel4;
            
            len++;
        }
        
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    printf("GPU Query time %0.2fms\n", time_kernel);
    printf("GPU query() time %0.2fms\n", time_kernel1);
    // printf("GPU getprfixsum() time %0.2fms\n", time_kernel2);
    // printf("GPU getnewqueue() time %0.2fms\n", time_kernel3);
    // int num;
    // for(int i = 0;i < offset[2];i++){
    //     printf("%d ",nodes_index_c[i]);
    //     printf("%d %d %d %d %d %d\n",i,nodes_mbr_c[i].x1,nodes_mbr_c[i].x2,nodes_mbr_c[i].y1,nodes_mbr_c[i].y2,key);
    // }
    // printf("\n");
    // return 0;
    long long sum = 0;
    long long max_num = 0;
    long long sum1 = 0;
    long long max_num1 = 0;
    long long sum2 = 0;
    long long max_num2 = 0;
    int num1;
    int num2;
    gettimeofday(&start, NULL);
    for(int i = 0;i < size;i++){
        num = 0;
        num2 = 0;
        num1 = RtreeSearch(nodes_index_c,nodes_mbr_c,0,nums[i],i,&num,num2);
        // printf("%d %d %d %d num = %d\n",nums[i].x1,nums[i].y1,nums[i].x2,nums[i].y2,num);
        max_num = max(max_num,(long long)num);
        max_num1 = max(max_num1,(long long)num1);
        max_num2 = max(max_num2,(long long)num2);
        sum += num;
        sum1 += num1;
        sum2 += num2;
    }
    // sum2 -= size;
    sum2 /= 2;
    gettimeofday(&end, NULL);
    elapsed = (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec;
    printf("qeury run: %0.2lfms\n",(double)elapsed / 1000);
    printf("sum = %ld  ave = %0.2lf max = %d\n",sum,(double)sum / size,max_num);
    printf("sum1 = %ld ave1 = %0.2lf max1 = %d \n",sum1,(double)sum1 / size,max_num1);
    printf("sum2 = %ld ave2 = %0.2lf max2 = %d \n",sum2,(double)sum2 / size,max_num2);
    sum = RtreeSearch1(nodes_index_c,nodes_mbr_c,0);
    printf("sum3 = %ld  ave3 = %0.2lf\n",sum,(double)sum / offset.back());
    return 0;
}

