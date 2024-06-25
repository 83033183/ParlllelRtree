#include <stdio.h>
#include<iostream>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
// #define m 10 // The minimum number of keys in each R-tree node.
#define m 10 // The minimum number of keys in each R-tree node.
#define M 19 // The maximum number of keys in each R-tree node.
#define LEAF 0
#define NONLEAF 1
#define INFINITE 100000000
#define NINFINITE -100000000
#define MINERROR 0.00000001

#define datatype bounding_box
#define BIT 4
#define BYTES 16

using namespace std;


typedef struct key{
	int index;
	unsigned long long int zOrderValue;
	//int zOrderValue;
}key;

//represents a bounding box.
typedef struct bounding_box{
	long long int rect[4];
	unsigned long long int x[4];
	unsigned long long int y[4];
	unsigned long long int zOrderValue;
	int data;
}bounding_box;

//represents a rectangle
typedef struct rtree_rectangle{
	long long int rect[4];
} rtree_rectangle; 

//represents a node or a leaf
//the structure might waste some space.
typedef struct rtree_node{
	int flag; //0 represents a node; 1 represents a leaf;
	int num; //the number of the node's children
	rtree_rectangle mbr[m];
	int children[m]; //only non leaf nodes need this field
	int data[m]; //only leaf nodes need this field
	int start,end;
} rtree_node;


__global__ void CalcuateZOrderValue(bounding_box * pointers,int num,int minX,int minY,long long int scale);
__device__ unsigned long long int Interleave(unsigned long long int a1,unsigned long long int a2,unsigned long long int a3,unsigned long long int a4);
__global__ void ParallelSorting(bounding_box * pointers,int num);
__global__ void Construction(bounding_box * pointers,int num,rtree_node * nodes,int * root);
__device__ void ResetMBR(rtree_rectangle * rect);
__device__ void Merge(rtree_rectangle source,rtree_rectangle target,rtree_rectangle *output);
int RtreeSearch(rtree_node * nodes,int num_nodes,rtree_rectangle target,int key,int* num,int &num1);
int RtreeSearchHelp(rtree_node *nodes,int index,rtree_rectangle target,int *num,int &num1,int key);
int Overlap(rtree_rectangle target,rtree_rectangle source);
void Merge1(rtree_rectangle source,rtree_rectangle target,rtree_rectangle *output);
__global__ void LocalSort(datatype *g_data,datatype *g_data_bak,int *histogram,int size,int round,int numOfBlocks);
__global__ void Reorder(datatype *g_data,datatype *g_data_out,int *histogram,int *position,int size,int round,int numOfBlocks);
__global__ void scan1(int *g_data,int *g_odata,int *output,int length,int size,int flag);
__global__ void scan2(int *g_data, int *output,int length,int size,int numOfBlocks);


__global__ void CalcuateZOrderValue(bounding_box * pointers,int num,int minX,int minY,long long int scale){
	int index=threadIdx.x;
	int bx=blockDim.x;
	int multiplier=1;
	int i;
	for (i= index; i < num; i=i+bx)
  	{ 
		
    	pointers[i].x[0]=(long long int )(pointers[i].rect[0] / scale *multiplier);
    	pointers[i].x[1]=(long long int )(pointers[i].rect[1] / scale *multiplier);
    	pointers[i].x[2]=(long long int )(pointers[i].rect[2] / scale *multiplier);
    	pointers[i].x[3]=(long long int )(pointers[i].rect[3] / scale *multiplier);
	}
	for (i= index; i < num; i=i+bx)
  	{
    	pointers[i].y[0]=(long long int )(pointers[i].x[0]-(long long int )(minX*multiplier));
    	pointers[i].y[1]=(long long int )(pointers[i].x[1]-(long long int )(minY*multiplier));
    	pointers[i].y[2]=(long long int )(pointers[i].x[2]-(long long int )(minX*multiplier));
    	pointers[i].y[3]=(long long int )(pointers[i].x[3]-(long long int )(minY*multiplier));
    	pointers[i].zOrderValue=Interleave(pointers[i].y[0],pointers[i].y[1],pointers[i].y[2],pointers[i].y[3]);
  	}
}

/* Calculate the z-order value for each bounding box using its four coordiantes*/
/* the algorithm is described at http://www-graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN */
__device__ unsigned long long int Interleave(unsigned long long int a1,unsigned long long int a2,unsigned long long int a3,unsigned long long int a4){
	
	unsigned long long int B[] = {0x0303030303030303, 0x000F000F000F000F, 0x000000FF000000FF};	
	unsigned int S[] = {6,12,24};
	unsigned long long int value;
               
	a1 = (a1 | (a1 << S[2])) & B[2];
	a1 = (a1 | (a1 << S[1])) & B[1];
	a1 = (a1 | (a1 << S[0])) & B[0];
	
	a2 = (a2 | (a2 << S[2])) & B[2];
	a2 = (a2 | (a2 << S[1])) & B[1];
	a2 = (a2 | (a2 << S[0])) & B[0];
	
	a3 = (a3 | (a3 << S[2])) & B[2];
	a3 = (a3 | (a3 << S[1])) & B[1];
	a3 = (a3 | (a3 << S[0])) & B[0];
	
	a4 = (a4 | (a4 << S[2])) & B[2];
	a4 = (a4 | (a4 << S[1])) & B[1];
	a4 = (a4 | (a4 << S[0])) & B[0];
	
	value= (a1<<6) | (a2 << 4)|(a3<<2)|a4;
	//printf("%llx\n",value);
	return value;
}

__global__ void LocalSort(datatype *g_data,datatype *g_data_bak,int *histogram,int size,int round,int numOfBlocks){
	__shared__ key temp[512]; //allocated on invocation 
	__shared__ int temp1[512];
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int base=bx*size;
	int thid = tx+base;
	int i;
	int pout = 0, pin = 1;
	int out=1,in=0;
	int zeros;
	int offset;
	temp[tx].zOrderValue = g_data[thid].zOrderValue;
	//temp[tx].zOrderValue=g_data[thid].data;
	temp[tx].index=tx;
	for(i=0;i<BIT;i++){
		//Count the number of zeros
		temp1[pout*size+tx]=(1-((temp[in*size+tx].zOrderValue>>(i+round))&0x00000001));
		__syncthreads();
		for (offset = 1; offset < size; offset *= 2) {
			pout = 1 - pout; // swap double buffer indices 
			pin = 1 - pout;
			if (tx >= offset) {
				temp1[pout*size+tx] = temp1[pin*size+tx-offset]+temp1[pin*size+tx];
			}
			else {
				temp1[pout*size+tx] = temp1[pin*size+tx];
			}
			__syncthreads(); 
		}
		temp[out*size+tx].index=temp1[pout*size+tx]*(1-(temp[in*size+tx].zOrderValue>>(i+round))&0x00000001);
		zeros=temp1[pout*size+size-1];
		//missing
		__syncthreads();
		temp1[pout*size+tx]=((temp[in*size+tx].zOrderValue>>(i+round))&0x00000001);
		__syncthreads();
		for (offset = 1; offset < size; offset *= 2) {
			pout = 1 - pout; // swap double buffer indices 
			pin = 1 - pout;
			if (tx >= offset) {
				temp1[pout*size+tx] = temp1[pin*size+tx-offset]+temp1[pin*size+tx];
			}
			else {
				temp1[pout*size+tx] = temp1[pin*size+tx];
			}
			__syncthreads(); 
		}
		temp1[pin*size+tx]=(temp1[pout*size+tx]+zeros)*((temp[in*size+tx].zOrderValue>>(i+round))&0x00000001)+temp[out*size+tx].index;
		__syncthreads(); 
		temp[out*size+(temp1[pin*size+tx]-1)]=temp[in*size+tx];
		out = 1 - out; // swap double buffer indices 
		in = 1 - out;
		__syncthreads(); 
	}
	g_data_bak[thid]=g_data[(temp[in*size+tx].index)+base];  
	for(i=0;i<BYTES;i++){
		if(((temp[in*size+tx].zOrderValue>>(round))&0x0000000F)==i){
			temp1[pin*size+tx]=1;
		}
		else{
			temp1[pin*size+tx]=0;
		}
		__syncthreads();
		//g_data[thid]=temp[pin*size+txx]; 
		int offset=size;		
		while(offset>1){
			offset=(offset>>1);
			if(tx<offset){
				temp1[pin*size+tx]=temp1[pin*size+tx+offset]+temp1[pin*size+tx];
			}
			__syncthreads();
		}
		if(tx==0){
			histogram[i*numOfBlocks+bx]=temp1[pin*size+tx];
		}
		__syncthreads(); 
	}
}
__global__ void Reorder(datatype *g_data,datatype *g_data_out,int *histogram,int *position,int size,int round,int numOfBlocks){
	__shared__ key temp[256]; // allocated on invocation 
	__shared__ int temp1[16];
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int base=bx*size;
	int thid = tx+base;
	temp[tx].zOrderValue = g_data[thid].zOrderValue;
	//temp[tx].zOrderValue = g_data[thid].data;
	temp[tx].index=thid;
	if(tx<BYTES){
		temp1[tx]=histogram[tx*numOfBlocks+bx];
	}
	if(tx==0){
		for(int i=1;i<BYTES;i++){
			temp1[i]=temp1[i]+temp1[i-1];
		}
	}
	__syncthreads();
	int index=(temp[tx].zOrderValue>>round)&0x0000000F;
	int pos=position[index*numOfBlocks+bx]-(temp1[index]-tx);
	g_data_out[pos]=g_data[temp[tx].index];
}
__global__ void scan1(int *g_data,int *g_odata,int *output,int length,int size,int flag) {  
	extern __shared__ int temp[]; // allocated on invocation 
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int base=bx*size;
	int thid = tx+base;
	int pout = 0, pin = 1;
	// load input into shared memory. 
	// This is exclusive scan, so shift right by one and set first elt to 0 
	//temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0; 
	if(thid<=length-1){ 
		temp[pout*size + tx] = g_data[thid];
		//temp[pin*n + thid] = 0.0;
		__syncthreads();
		for (int offset = 1; offset < size; offset *= 2) {
			pout = 1 - pout; // swap double buffer indices 
			pin = 1 - pout;
			if (tx >= offset) {
				temp[pout*size+tx] = temp[pin*size+tx - offset]+temp[pin*size+tx];
				//temp[pout*n+thid] += temp[pin*n+thid - offset];
			}
			else {
				temp[pout*size+tx] = temp[pin*size+tx];
			}
			__syncthreads(); 
		}
		g_odata[thid] = temp[pout*size+tx]; // write output
		if((tx==size-1) && flag) {
			output[bx]=temp[pout*size+tx];
		}
	}
}
__global__ void scan2(int *g_data, int *output,int length,int size,int numOfBlocks) {
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int base=bx*size;
	int thid = tx+base;
	if((thid<=length-1)&&(bx!=0)){ 
		g_data[thid]=g_data[thid]+output[bx-1];	
	}
}



//parallel sorting (odd even sorting)
__global__ void ParallelSorting(bounding_box * pointers,int num)
{
		int i;
		int index=threadIdx.x;
		int bx=blockDim.x;
		bounding_box swap;
		__shared__ int done[2];
		done[0]=0;
		done[1]=0;
		int count=0;
		int oddCycle=0;
		//for(j=0;j<=num;j++){
		do{		
	        //done[0]=1;
	        //done[1]=1;
	        done[oddCycle]=1;
	        //at the end of the array, we add one dummy element to avoid branching.
	        for (i = index*2+oddCycle; i < num; i=i+bx*2) 
    	    {
	 			if(pointers[i].zOrderValue>pointers[i+1].zOrderValue){
	 				swap=pointers[i+1];
	 				pointers[i+1]=pointers[i];
	 				pointers[i]=swap;
	 				done[oddCycle]=0;
	 			}
        	} //for
        	oddCycle = !(oddCycle);
        	count++;
         	__syncthreads();
		//}
		}while((!done[0])||(!done[1]));
}

/* Implement the bottom-up R-tree construction */
__global__ void Construction(bounding_box * pointers,int num,rtree_node * nodes,int * root){
	int i,j,k;
	int numOfNodes;
	int numOfGroup;
	int numOfChildren;
	int index,childrenIndex;
	int begin,end;
	int tx=threadIdx.x;
	int bx=blockDim.x;
	long long int temp;
	index=0;
	numOfNodes=num;
	numOfGroup=numOfNodes/ m;	
	for(i=tx;i<numOfGroup;i=i+bx){
		temp=numOfNodes;
		temp=temp*i;
		begin=(int)(temp/numOfGroup);
		if(begin<0){
			childrenIndex=i;
		}
		temp=numOfNodes;
		temp=temp*(i+1);
		end=(int)(temp/numOfGroup-1);
		nodes[i].flag=LEAF;
		nodes[i].num=(end-begin+1);
		for(j=begin;j<=end;j++){
			nodes[i].data[j-begin]=pointers[j].data;
			nodes[i].mbr[j-begin].rect[0]=pointers[j].rect[0];
			nodes[i].mbr[j-begin].rect[1]=pointers[j].rect[1];
			nodes[i].mbr[j-begin].rect[2]=pointers[j].rect[2];
			nodes[i].mbr[j-begin].rect[3]=pointers[j].rect[3];
			nodes[i].children[j-begin]=-1;
		}
	}
	index=index+numOfGroup;
	numOfChildren=numOfGroup;
	childrenIndex=0;
	__syncthreads();
	while(numOfGroup>1){
		numOfNodes=numOfGroup;
		numOfGroup=numOfNodes/ m;
		for(i=index+tx;i<(numOfGroup+index);i=i+bx){
			temp=numOfNodes;
			temp=temp*(i-index);
			begin=(int)(temp/numOfGroup+childrenIndex);
			temp=numOfNodes;
			temp=temp*(i+1-index);
			end=(int)(temp/numOfGroup-1+childrenIndex);
			nodes[i].flag=NONLEAF;
			nodes[i].num=(end-begin+1);
			for(j=begin;j<=end;j++){
				nodes[i].children[j-begin]=j;
				nodes[i].data[j-begin]=-1;
				ResetMBR(&(nodes[i].mbr[j-begin]));
				for(k=0;k<nodes[j].num;k++){
					Merge(nodes[i].mbr[j-begin],nodes[j].mbr[k],&(nodes[i].mbr[j-begin]));
				}

			}
		}
		__syncthreads();
		index=index+numOfGroup;
		childrenIndex=childrenIndex+numOfChildren;
		numOfChildren=numOfGroup;
	}
	(*root)=index-numOfGroup;
}

__global__ void Construction_init(bounding_box * pointers,int num,rtree_node * nodes,int * root){
	int i,j,k;
	int numOfNodes;
	int numOfGroup;
	int numOfChildren;
	int index,childrenIndex;
	int begin,end;
	int tx= threadIdx.x + blockIdx.x * blockDim.x;
	int bx= blockDim.x * gridDim.x;
	long long int temp;
	index=0;
	numOfNodes=num;
	numOfGroup=numOfNodes / m + (numOfNodes % m != 0);	
	for(i=tx;i<numOfGroup;i=i+bx){
		begin=i * m;
		end=min((i * m + m -1),numOfNodes - 1);;
		nodes[i].flag=LEAF;
		nodes[i].num=(end-begin+1);
		nodes[i].strat = begin;
		nodes[i].end = end;
		for(j=begin;j<=end;j++){
			nodes[i].data[j-begin]=pointers[j].data;
			nodes[i].mbr[j-begin].rect[0]=pointers[j].rect[0];
			nodes[i].mbr[j-begin].rect[1]=pointers[j].rect[1];
			nodes[i].mbr[j-begin].rect[2]=pointers[j].rect[2];
			nodes[i].mbr[j-begin].rect[3]=pointers[j].rect[3];
			nodes[i].children[j-begin]=-1;
		}
	}
}

__global__ void Construction1(bounding_box * pointers,int num,rtree_node * nodes,int * root,int numOfGroup,int numOfNodes,int index,int childrenIndex,int numOfChildren){
	int i,j,k;
	int begin,end;
	int tx= threadIdx.x + blockIdx.x * blockDim.x;
	int bx= blockDim.x * gridDim.x;
	long long int temp;
	for(i=index+tx;i<(numOfGroup+index);i=i+bx){
		begin=(i - index) * m +childrenIndex;
		end=min((i - index) * m + m - 1 +childrenIndex,numOfNodes+childrenIndex - 1);
		nodes[i].flag=NONLEAF;
		nodes[i].num=(end-begin+1);
		nodes[i].start = nodes[begin].start;
		nodes[i].end = nodes[end].end;
		for(j=begin;j<=end;j++){
			nodes[i].children[j-begin]=j;
			nodes[i].data[j-begin]=-1;
			ResetMBR(&(nodes[i].mbr[j-begin]));
			for(k=0;k<nodes[j].num;k++){
				Merge(nodes[i].mbr[j-begin],nodes[j].mbr[k],&(nodes[i].mbr[j-begin]));
			}
		}
	}
}
/***********************************************************************/
/*  FUNCTION:  RtreeSearch  */
/**/
/*  INPUTS: The pointer to the rtree, target rectangle, the pointer to the output info array*/
/**/
/*  OUTPUT: The number of the nodes whose rectangles overlap with target rectangle */
/**/
/*  Modifies input:  the pointer to the output info array */
/**/
/*  Effect:  Search the rtree nodes whose rectangles overlap with the target rectangle. This function */
/*  also calls  RtreeSearchHelp function */
/***********************************************************************/
int RtreeSearch(rtree_node * nodes,int num_nodes,rtree_rectangle target,int key,int * num,int &num1){
	(*num) = 0;
    return RtreeSearchHelp(nodes,num_nodes,target,num,num1,key);
}

/***********************************************************************/
/*  FUNCTION:  RtreeSearchHelp  */
/**/
/*  INPUTS: A pointer to the parent node, a target rectangle, and the number of related rtree nodes */
/**/
/*  OUTPUT: None */
/**/
/*  Modifies input:  the number of relatd rtree nodes */
/**/
/*  Effect:  Recursively search the children node  */
/***********************************************************************/
int RtreeSearchHelp(rtree_node *nodes,int index,rtree_rectangle target,int *num,int& num1,int key){
	int i;
    int ans = -1;
	if(nodes[index].flag==NONLEAF){
		for(i=0;i<nodes[index].num;i++){
			(*num)++;
			// printf("%d(%d,%d,%d,%d), comparing with node[%d].mbr[%d] (%d,%d,%d,%d)\n",key,target.rect[0],target.rect[1],target.rect[2],target.rect[3],index,i,nodes[index].mbr[i].rect[0],nodes[index].mbr[i].rect[1],nodes[index].mbr[i].rect[2],nodes[index].mbr[i].rect[3]);
			if(Overlap(target,nodes[index].mbr[i])){
					//printf("overlap\n");
                    ans++;
					ans += RtreeSearchHelp(nodes,nodes[index].children[i],target,num,num1,key);
			}
		}
	}
	else{
		for(i=0;i<nodes[index].num;i++){
	  		*num=(*num)+1;
	  		// if(nodes[index].data[i]==key){
		  		// if(nodes[index].num > 1)
				// printf("num = %d\n",nodes[index].num);
		  	// }
			if(Overlap(target,nodes[index].mbr[i])){
				ans++;
				num1++;
			}
	  	}
	}
    return max(0,ans);
}

/***********************************************************************/
/*  FUNCTION:  Overlap  */
/**/
/*  INPUTS: Two rectangles */
/**/
/*  OUTPUT: 1: overlapped; 0: not overlapped */
/**/
/*  Modifies input:  None */
/**/
/*  Effect: Determine if two rectangles overlap */
/***********************************************************************/
int Overlap(rtree_rectangle target,rtree_rectangle source){
	rtree_rectangle output;
	Merge1(target,source,&output);
	if(((unsigned long long int)target.rect[2]+source.rect[2]-target.rect[0]-source.rect[0])+MINERROR<(output.rect[2]-output.rect[0])){
		return 0;
	}
	if(((unsigned long long int)target.rect[3]+source.rect[3]-target.rect[1]-source.rect[1])+MINERROR<(output.rect[3]-output.rect[1])){
		return 0;
	}
	return 1;
}

/***********************************************************************/
/*  FUNCTION:  Merge  */
/**/
/*  INPUTS: Two rectangles to be merged and the output rectangle */
/**/
/*  OUTPUT: None */
/**/
/*  Modifies input:  Output rectangle */
/**/ 
/*  Effect:  Merge two rectangles and form a new rectangle */
/***********************************************************************/
void Merge1(rtree_rectangle source,rtree_rectangle target,rtree_rectangle *output){
	output->rect[0]=(source.rect[0]<target.rect[0])?source.rect[0]:target.rect[0];
	output->rect[1]=(source.rect[1]<target.rect[1])?source.rect[1]:target.rect[1];
	output->rect[2]=(source.rect[2]>target.rect[2])?source.rect[2]:target.rect[2];
	output->rect[3]=(source.rect[3]>target.rect[3])?source.rect[3]:target.rect[3];
}

/***********************************************************************/
/*  FUNCTION:  ResetMBR  */
/**/
/*  INPUTS: A rectangle */
/**/
/*  OUTPUT: None */
/**/
/*  Modifies input:  None */
/**/
/*  Effect: Initiate a MBR */
/***********************************************************************/
__device__ void ResetMBR(rtree_rectangle * rect){
	rect->rect[0]=INFINITE;
	rect->rect[1]=INFINITE;
	rect->rect[2]=NINFINITE;
	rect->rect[3]=NINFINITE;
}

/***********************************************************************/
/*  FUNCTION:  Merge  */
/**/
/*  INPUTS: Two rectangles to be merged and the output rectangle */
/**/
/*  OUTPUT: None */
/**/
/*  Modifies input:  Output rectangle */
/**/
/*  Effect:  Merge two rectangles and form a new rectangle */
__device__ void Merge(rtree_rectangle source,rtree_rectangle target,rtree_rectangle *output){
	output->rect[0]=(source.rect[0]<target.rect[0])?source.rect[0]:target.rect[0];
	output->rect[1]=(source.rect[1]<target.rect[1])?source.rect[1]:target.rect[1];
	output->rect[2]=(source.rect[2]>target.rect[2])?source.rect[2]:target.rect[2];
	output->rect[3]=(source.rect[3]>target.rect[3])?source.rect[3]:target.rect[3];
}


__device__ int Overlap1(rtree_rectangle target,rtree_rectangle source){
	rtree_rectangle output;
	Merge(target,source,&output);
	if(((unsigned long long int)target.rect[2]+source.rect[2]-target.rect[0]-source.rect[0])+MINERROR<(output.rect[2]-output.rect[0])){
		return 0;
	}
	if(((unsigned long long int)target.rect[3]+source.rect[3]-target.rect[1]-source.rect[1])+MINERROR<(output.rect[3]-output.rect[1])){
		return 0;
	}
	return 1;
}

int RtreeSearch1(rtree_node *nodes,int index){
	int i;
    int ans = 0;
	for(i=1;i<nodes[index].num;i++){
		for(int j = i - 1;j >= 0;j--){
			if(Overlap(nodes[index].mbr[j],nodes[index].mbr[i]))
                ans++;
		}
	}

	if(nodes[index].flag==NONLEAF){
		for(i = 0;i < nodes[index].num;i++){
			ans += RtreeSearch1(nodes,nodes[index].children[i]);
		}
	}
    return ans;
}

__global__ void query1(rtree_node * nodes,int * query_qeue,int * query_qeue_len,rtree_rectangle * targets,int querysize,int queryarrsize){
    int inx = threadIdx.x;
    int bix = blockIdx.x;
    int threadSize = blockDim.x;
    int blockSize = gridDim.x;
    int size;
    rtree_rectangle target;
    for(int i = bix;i < querysize;i+= blockSize){
        size = query_qeue_len[i];
        __syncthreads();
        if(inx == 0)
            query_qeue_len[i] = 0;
        __syncthreads();
        target = targets[i];
        for(int j = inx;j < size;j += threadSize){
            for(int k = 0;k < nodes[query_qeue[i * queryarrsize + j]].num;k++){
                if(Overlap1(target,nodes[query_qeue[i * queryarrsize + j]].mbr[k]) && nodes[query_qeue[i * queryarrsize + j]].flag == NONLEAF){
                    query_qeue[i * queryarrsize + atomicAdd(&query_qeue_len[i],1)] = nodes[query_qeue[i * queryarrsize + j]].children[k];
                }
            }
        }
    }
}

__global__ void init(bounding_box * nums,int numsSize,int offset,int * query_qeue,int *query_qeue_len,rtree_rectangle * targets,int querysize,int queryarrsize){
    int inx = threadIdx.x + blockDim.x * blockIdx.x;
    int threadSize = blockDim.x * gridDim.x;
    for(int i = inx;i < querysize;i += threadSize){
        query_qeue[i * queryarrsize] = 0;
        if(i + offset < numsSize){
			query_qeue_len[i] = 1;
			for(int j = 0;j < 4;j++)
				targets[i].rect[j] = nums[i + offset].rect[j];
		}else
			query_qeue_len[i] = 0;
        // if(inx < 1)
        //     printf("%d %d %d %d\n",nums[i + offset].x1,nums[i + offset].y1,nums[i + offset].x2,nums[i + offset].y2);
    }
}

void ResetMBR1(rtree_rectangle * rect){
	rect->rect[0]=INFINITE;
	rect->rect[1]=INFINITE;
	rect->rect[2]=NINFINITE;
	rect->rect[3]=NINFINITE;
}
int cmp(const void *a,const void *b){
    return (*(bounding_box*)(a)).zOrderValue > (*(bounding_box*)(b)).zOrderValue;
}

int main(int argc , char **argv){

	FILE *fp;
  	bounding_box * pointers;
  	bounding_box * mbrs;
  	bounding_box * mbrs_bak;
  	rtree_node * nodes;
  	rtree_node * nodes_gpu;
  	int num_contours;
  	int num_nodes;
  	long long int x1,y1,x2,y2;
  	long long int minX,maxX,minY,maxY;
  	int i,j;
  	int key=0;
  	int root;
  	int sum=0;
  	int * root_gpu;
  	unsigned long long int maxZOrderValue=0xffffffffffffffff; 
  	//bounding_box temp;
  	//temp.zOrderValue=maxZOrderValue;

	minX=1e17;
	maxX=0;
	minY=1e17;
	maxY=0;
	
    // 4194304
  	fp=fopen("data/Rect1024.txt","r"); 
  	//fp=fopen("sample.txt","r");
  	fscanf(fp, "%d", &num_contours);
   	//printf("%d\n",num_contours);
	// num_contours = min(num_contours,10000);
   	num_nodes=3*num_contours/m;
    pointers=(bounding_box *)malloc(num_contours*sizeof(bounding_box));
    nodes=(rtree_node *)malloc(num_nodes*sizeof(rtree_node));
  	for (i= 0; i < num_contours; i++) 
  	{
    	fscanf(fp, "%lld %lld", &x1,&x2);
    	fscanf(fp, "%lld %lld", &y1,&y2);
    	fscanf(fp, "%d", &key);
		if(x1<minX) {
			minX=x1;
		}
		if(x2>maxX){
			maxX=x2;
		}
		if(y1<minY){
			minY=y1;
		}
		if(y2>maxY){
			maxY=y2;
		}
		pointers[i].rect[0]=x1;
    	pointers[i].rect[1]=y1;
    	pointers[i].rect[2]=x2;
    	pointers[i].rect[3]=y2;
    	pointers[i].data=key;

		// if(i < 10)
		// 	printf("%lld %lld %lld %lld\n",x1,x2,y1,y2);
    }
	printf("%lld %lld %lld %lld\n",minX,maxX,minY,maxY);
	for (i= 0; i < num_contours; i++) {
		pointers[i].rect[0]= (pointers[i].rect[0]- minX) / 1;
		pointers[i].rect[1]= (pointers[i].rect[1]- minY) / 1;
		pointers[i].rect[2]= (pointers[i].rect[2]- minX) / 1;
		pointers[i].rect[3]= (pointers[i].rect[3]- minY) / 1;
	}
	
	maxX = max(maxX - minX,maxY - minY);
	maxY = 1;
	while(maxX > 1e4){
		maxY *= 10;
		maxX /= 10;
	}   
	printf("%lld\n",maxY);
	// // minX /= maxY;
	// // minY /= maxY;
	minX = 0;
	minY = 0;
    /******************************/
    int length=num_contours;
    int size=256;
    int numOfBlocks=(length-1)/size+1;
    //printf("numOfBlocks=%d\n",numOfBlocks);
    int len=numOfBlocks*size;
	bounding_box * packs=(bounding_box *)malloc((len-length)*sizeof(bounding_box));
	for(i=0;i<(len-length);i++){
		packs[i].zOrderValue=maxZOrderValue;
	}

    int* histogram=0;
    int* position=0;

    (cudaMalloc( (void**) &histogram, numOfBlocks*16*sizeof(int)));
    (cudaMalloc( (void**) &position, numOfBlocks*16*sizeof(int)));
	
	int firstBlocks=(numOfBlocks*BYTES-1)/512+1;
	int* firstOutput=0;
	(cudaMalloc( (void**) &firstOutput, firstBlocks*sizeof(datatype)));
    /******************************/
    
    //allocate device memory
    (cudaMalloc( (void**) &nodes_gpu, sizeof(rtree_node)*num_nodes));
   	(cudaMalloc( (void**) &mbrs, sizeof(bounding_box)*len));
   	(cudaMalloc( (void**) &mbrs_bak, sizeof(bounding_box)*len));
    (cudaMalloc( (void**) &root_gpu, sizeof(int)));
	//checkErrors("Memory allocation");
    //copy data from memory to device memory
    (cudaMemcpy( mbrs, pointers, sizeof(bounding_box)*num_contours, cudaMemcpyHostToDevice));
    (cudaMemcpy( (mbrs+num_contours), packs,sizeof(bounding_box)*(len-length),cudaMemcpyHostToDevice));
    
	//(cudaMemcpy( (mbrs+num_contours), &temp, sizeof(bounding_box), cudaMemcpyHostToDevice));
    //checkErrors("Memory copy 1");
    dim3 dimBlock(512,1);
    dim3 dimGrid(1,1);
    
    dim3 dimBlockSort(size,1,1);
    dim3 dimGridSort(numOfBlocks,1);
	
	dim3 dimBlockScan(512,1,1);
    dim3 dimGridScan(firstBlocks,1);
    dim3 singleGrid(1,1);
    //timing	
	

	//Call kernel(global function)
	CalcuateZOrderValue<<<dimGrid, dimBlock>>>(mbrs,num_contours,minX,minY,maxY);
	cudaThreadSynchronize();




	(cudaMemcpy( pointers, mbrs, sizeof(bounding_box)*num_contours, cudaMemcpyDeviceToHost));
	struct timeval start, end;
    long elapsed;
    gettimeofday(&start, NULL);

	qsort(pointers,num_contours,sizeof(bounding_box),cmp);

	int numOfNodes;
	int numOfGroup;
	int numOfChildren;
	int index,childrenIndex;
	int begin,end1;
	long long int temp2;
	index=0;
	numOfNodes=num_contours; 
	numOfGroup=numOfNodes / m + (numOfNodes % m != 0);
	for(int i = 0;i < numOfGroup;i++){
		begin=i * m;
		end1=min((i * m + m -1),numOfNodes - 1);
		nodes[i].flag=LEAF;
		nodes[i].num=(end1-begin+1);
		for(int j=begin;j<=end1;j++){
			nodes[i].data[j-begin]=pointers[j].data;
			nodes[i].mbr[j-begin].rect[0]=pointers[j].rect[0];
			nodes[i].mbr[j-begin].rect[1]=pointers[j].rect[1];
			nodes[i].mbr[j-begin].rect[2]=pointers[j].rect[2];
			nodes[i].mbr[j-begin].rect[3]=pointers[j].rect[3];
			nodes[i].children[j-begin]=-1;
		}
	}
	index=index+numOfGroup;
	numOfChildren=numOfGroup;
	childrenIndex=0;
	while(numOfGroup > 1){
		numOfNodes=numOfGroup;
		numOfGroup=numOfNodes / m + (numOfNodes % m != 0);
		
		for(int i=index;i<(numOfGroup+index);i++){
			begin=(i - index) * m +childrenIndex;
			end1=min((i - index) * m + m - 1 +childrenIndex,numOfNodes+childrenIndex - 1);
			nodes[i].flag=NONLEAF;
			nodes[i].num=(end1-begin+1);
			for(int j=begin;j<=end1;j++){
				nodes[i].children[j-begin]=j;
				nodes[i].data[j-begin]=-1;
				ResetMBR1(&(nodes[i].mbr[j-begin]));
				for(int k=0;k<nodes[j].num;k++){
					Merge1(nodes[i].mbr[j-begin],nodes[j].mbr[k],&(nodes[i].mbr[j-begin]));
				}
			}
		}

		index=index+numOfGroup;
		childrenIndex=childrenIndex+numOfChildren;
		numOfChildren=numOfGroup;
	}
	 gettimeofday(&end, NULL);
    elapsed = (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec;
    printf("CPU Construction: %0.2lfms\n",(double)elapsed / 1000);


	cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);
	///*
	for(i=0;i<16;i++){
		LocalSort<<<dimGridSort, dimBlockSort>>>(mbrs,mbrs_bak,histogram,size,i*4,numOfBlocks);
		scan1<<<dimGridScan, dimBlockScan,(2*512*sizeof(int))>>>(histogram,position,firstOutput,numOfBlocks*BYTES,512,1);
		if(firstBlocks>1){
			scan1<<<singleGrid, dimBlockScan,(2*512*sizeof(int))>>>(firstOutput,firstOutput,firstOutput,firstBlocks,512,0);
    		scan2<<<dimGridScan, dimBlockScan>>>(position,firstOutput,len,512,firstBlocks);
		}
		Reorder<<<dimGridSort, dimBlockSort>>>(mbrs_bak,mbrs,histogram,position,size,i*4,numOfBlocks);
	}

	Construction_init<<<32,512>>>(mbrs,num_contours,nodes_gpu,root_gpu);
	
	index=0;
	numOfNodes=num_contours; 
	numOfGroup=numOfNodes / m + (numOfNodes % m != 0);
	index=index+numOfGroup;
	numOfChildren=numOfGroup;
	childrenIndex=0;
	while(numOfGroup > 1){
		numOfNodes=numOfGroup;
		numOfGroup=numOfNodes/ m + (numOfNodes % m != 0);
		Construction1<<<32,512>>>(mbrs,num_contours,nodes_gpu,root_gpu,numOfGroup,numOfNodes,index,childrenIndex,numOfChildren);
		index=index+numOfGroup;
		childrenIndex=childrenIndex+numOfChildren;
		numOfChildren=numOfGroup;
	}
	int * temp1 = (int*)malloc(sizeof(int));
	(*temp1)=index-numOfGroup;
	cudaMemcpy(root_gpu,temp1,sizeof(int),cudaMemcpyHostToDevice);
	cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);	
	float time_kernel; 
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
	printf("GPU Construction %0.2f\n", time_kernel);

	//printf("start Sorting\n");
	//ParallelSorting<<<dimGrid, dimBlock>>>(mbrs,num_contours);
	// (cudaMemcpy(pointers,mbrs_bak, sizeof(bounding_box)*num_contours, cudaMemcpyDeviceToHost));
	(cudaMemcpy(pointers,mbrs, sizeof(bounding_box)*num_contours, cudaMemcpyDeviceToHost));
	/*for(i=1;i<num_contours;i++){
		if(pointers[i].zOrderValue<pointers[i-1].zOrderValue){
			printf("%d\t\t%lu\n",pointers[i].data,pointers[i].zOrderValue);	
		}
		//printf("%d\t%d\n",i,pointers[i].data);	
	}*/	
	//printf("start construction\n"); 

    
	//copy data from device memory to memory
	(cudaMemcpy(nodes,nodes_gpu, sizeof(rtree_node)*num_nodes, cudaMemcpyDeviceToHost));
	(cudaMemcpy(&root,root_gpu, sizeof(int), cudaMemcpyDeviceToHost));
    //checkErrors("Memory copy 2"); 
	///*
    // for(i=0;i<=root;i++){
	//   	for(j=0;j<nodes[i].num;j++){
	// 	  	printf("%d %d ",i,nodes[i].num);
	//   		if(nodes[i].flag==LEAF){
	//   			printf("*%d*",nodes[i].data[j]);
	//   			printf(" (%d,%d,%d,%d) ",nodes[i].mbr[j].rect[0],nodes[i].mbr[j].rect[1],nodes[i].mbr[j].rect[2],nodes[i].mbr[j].rect[3]);
	//   		}
	//   		else{
	//   			printf("[%d]",nodes[i].children[j]); 
	//   			printf(" (%d,%d,%d,%d) ",nodes[i].mbr[j].rect[0],nodes[i].mbr[j].rect[1],nodes[i].mbr[j].rect[2],nodes[i].mbr[j].rect[3]);
	//   		}
	//   		printf("\n");
	//   	}
	//   }
	//*/
	//printf("start searching\n");
	cudaFree(mbrs_bak);
	cudaFree(histogram);
	cudaFree(position);

	int queryarrsize = 1000;
    int querysize = 10000;
    int * query_qeue;
    int * query_qeue_len;
    // int * prfix;
    rtree_rectangle * targets;
    cudaMalloc((void**)& query_qeue,sizeof(int) * querysize * queryarrsize);
    // cudaMalloc((void**)& prfix,sizeof(int) * querysize * queryarrsize);
    cudaMalloc((void**)& query_qeue_len,sizeof(int) * querysize);
    cudaMalloc((void**)& targets,sizeof(rtree_rectangle) * querysize);
    bool * flag_g;
    cudaMalloc((void**)& flag_g,sizeof(bool));
    bool flag = true;
    // int len;
    float time_kernel1 = 0,time_kernel2 = 0,time_kernel3 = 0,time_kernel4;
	rtree_rectangle temp; 
	int offsetsize = 0;
	int cnt = num_contours;
	while(cnt > 1){
		offsetsize++;
		cnt /= m;
	}
    cudaEvent_t start_event1, stop_event1;
    cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);

    for(int i = 0;i < num_contours;i += querysize){
        
		init<<<128,512>>>(mbrs,num_contours,i,query_qeue,query_qeue_len,targets,querysize,queryarrsize);
		len = 0;
        flag = true;
        while(len <  offsetsize){
            // printf("%d \n",len);
            // query1<<<8,512>>>(nodes_index,nodes_mbr,query_qeue,query_qeue_len,targets,querysize,queryarrsize);
            // check<<<1,1>>>(flag_g,query_qeue_len,querysize);
            // cudaMemcpy(&flag,flag_g,sizeof(bool),cudaMemcpyDeviceToHost);
            cudaEventCreate(&start_event1) ;
            cudaEventCreate(&stop_event1) ;
            cudaEventRecord(start_event1, 0);
            // query<<<16,512>>>(nodes_index,nodes_mbr,query_qeue,query_qeue_len,targets,prfix,querysize,queryarrsize);
            query1<<<8,512>>>(nodes_gpu,query_qeue,query_qeue_len,targets,querysize,queryarrsize);
            cudaEventRecord(stop_event1, 0);
            cudaEventSynchronize(stop_event1);
            cudaEventElapsedTime(&time_kernel4, start_event1, stop_event1);
            time_kernel1 += time_kernel4;

            // cudaEventCreate(&start_event1) ;
            // cudaEventCreate(&stop_event1) ;
            // cudaEventRecord(start_event1, 0);
            // getprifixsum<<<16,512,sizeof(int) * queryarrsize>>>(query_qeue_len,prfix,querysize,queryarrsize);
            // cudaEventRecord(stop_event1, 0);
            // cudaEventSynchronize(stop_event1);
            // cudaEventElapsedTime(&time_kernel4, start_event1, stop_event1);
            // time_kernel2 += time_kernel4;

            // cudaEventCreate(&start_event1) ;
            // cudaEventCreate(&stop_event1) ;
            // cudaEventRecord(start_event1, 0);
            // getnewqueue<<<16,512,sizeof(int) * queryarrsize>>>(nodes_index,query_qeue,query_qeue_len,prfix,querysize,queryarrsize);
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
	// printf("GPU cudaMemcpy() time %0.2fms\n", time_kernel2);

	int max_num = 0;
    int max_num1 = 0;
	int max_num2 = 0;
	long long sum2 = 0;
    long long sum1 = 0;
    int num;
	int num2;
	unsigned long long int MAX = (unsigned long long int)pow(2,32);
    gettimeofday(&start, NULL);
	for(i=0;i<num_contours - num_contours + num_contours;i++){
		rtree_rectangle temp; 
		int count;
		num2 = 0;
		temp.rect[0]=pointers[i].rect[0];
		temp.rect[1]=pointers[i].rect[1];
		temp.rect[2]=pointers[i].rect[2];
		temp.rect[3]=pointers[i].rect[3];
		count=RtreeSearch(nodes,root,temp,pointers[i].data,&num,num2);
		// printf("%d %d %d\n",i,count,root);
		// printf(" (%d,%d,%d,%d)\n ",pointers[i].rect[0],pointers[i].rect[1],pointers[i].rect[2],pointers[i].rect[3]);
		max_num1 = max(max_num1,count);
        max_num = max(max_num,num);
		max_num2 = max(max_num2,num2);
		sum1+=count;
        sum += num;  
		sum2 += num2;
	}
	// sum2 -= num_contours;
	sum2 /= 2;
    gettimeofday(&end, NULL);
    elapsed = (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec;
    
    printf("qeury run: %0.2lfms\n",(double)elapsed / 1000);
	printf("sum=%ld  ave = %0.2lf max = %d\n",sum,(double)sum / num_contours,max_num);
	printf("sum1=%ld  ave = %0.2lf max1 = %d\n",sum1,(double)sum1 / num_contours,max_num1);
    printf("sum2=%ld  ave = %0.2lf max2 = %d\n",sum2,(double)sum2 / num_contours,max_num2);
	sum = RtreeSearch1(nodes,root);
	printf("sum3 = %ld  ave3 = %0.2lf\n",sum,(double)sum / num_nodes / m);
	return 0;
}

