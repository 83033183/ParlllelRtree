#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include "gpc.h"
#include "common.h"
//#include "util.h"

/* Horizontal edge state transitions within scanbeam boundary */
__constant__  h_state next_h_state[3][6];

void printfOutput(gpc_polygon *result,int numOfOutput){
	int i,j;
	int count1=0;
	int count2=0;
	gpc_polygon * result_cpu;
	gpc_vertex_list *contour;
	gpc_vertex *vertex;
	cudaError_t err;

	MALLOC_CPU(result_cpu, numOfOutput * sizeof(gpc_polygon),"output polygon creation", gpc_polygon);
   	err=cudaMemcpy(result_cpu,result,sizeof(gpc_polygon)*numOfOutput,cudaMemcpyDeviceToHost);	 
   	if( err != cudaSuccess)
	{
		printf("CUDA error a: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
   	//MALLOC(contour, result_cpu[0].num_contours * sizeof(gpc_vertex_list),"contour creation", gpc_vertex_list);
   	//printf("(%d,%d)\n",sizeof(gpc_vertex_list)*result_cpu[0].num_contours,result_cpu[0].contour);
   	//err=cudaMemcpy(contour,result_cpu[0].contour,sizeof(gpc_vertex_list)*result_cpu[0].num_contours,cudaMemcpyDeviceToHost);	 
   	//MALLOC(contour,sizeof(gpc_vertex_list),"contour creation", gpc_vertex_list);
   	//err=cudaMemcpy(contour,result_cpu[1].contour,sizeof(gpc_vertex_list),cudaMemcpyDeviceToHost);	 
   	//if( err != cudaSuccess)
	//{
	//	printf("CUDA error b: %s\n", cudaGetErrorString(err));
	//	exit(-1);
	//}
   	//MALLOC(vertex, contour[0].num_vertices * sizeof(gpc_vertex),"vertex creation printf", gpc_vertex);
   	//cudaMemcpy(vertex,contour[0].vertex,sizeof(gpc_vertex)*contour[0].num_vertices,cudaMemcpyDeviceToHost);	 

	//printf("The %d contour has %d vertices\n",2,contour[0].num_vertices);

   	for(i=0;i<numOfOutput;i++){
		//printf("The %d output polygon has %d contours\n",(i+1),result_cpu[i].num_contours);
		if(result_cpu[i].num_contours>0){
			count1++;
		}
		//for(j=0;j<contour[0].num_vertices;j++){
			//printf("(%f,%f)\n",vertex[j].x,vertex[j].y);
			//printf("The %d contour has %d vertices\n",(j+1),result_cpu[i].contour[j].num_vertices);
		//}
		//printf("***********************************\n");
		//printf("The %d contour has %d vertices\n",1,contour[0].num_vertices);
   	}

   	printf("count1=%d,count2=%d\n",count1,count2);
}
 
void run(rtree_search *output_after, int numOfNodes,int * num_base,long int * prefix_base,gpc_vertex * vertex_base,int *num_overlay,long int * prefix_overlay, gpc_vertex * vertex_overlay,int numOfOutput,int *hole_base,int *hole_overlay,bounding_box *zOrder){
    h_state arr[3][6]={{BH, TH,   TH, BH,   NH, NH}, {NH, NH,   NH, NH,   TH, TH}, {NH, NH,   NH, NH,   BH, BH}};
	cudaMemcpyToSymbol(next_h_state, arr, 18*sizeof(h_state), 0, cudaMemcpyHostToDevice);
    gpc_polygon *result;
    cudaMalloc((void***)&result,sizeof(gpc_polygon)*numOfOutput);
    dim3 dimBlockExecute(128,1);
	dim3 dimGridExecute((numOfOutput-1)/128+1,1);
	//dim3 dimBlockExecute(117,1);//107 96 117
	//dim3 dimGridExecute(1,1);
	//__global__ void Execute(rtree_search *output,int offset,int * num_base,int * prefix_base,gpc_vertex * vertice_base,int *num_overlay,int * prefix_overlay, gpc_vertex * vertice_overlay,int num,gpc_polygon * result){
    size_t size;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize,1073741824);
    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
    printf("Heap size found to be %dn",(int)size);
    xmcInit(); 
    Execute<<<dimGridExecute,dimBlockExecute>>>(output_after,numOfNodes,num_base,prefix_base,vertex_base,num_overlay,prefix_overlay,vertex_overlay,numOfOutput,result,hole_base,hole_overlay,zOrder);
	printfOutput(result,numOfOutput);
	checkErrors("Memory check 1");
}

__global__ void Execute(rtree_search *output,int offset,int * num_base,long int * prefix_base,gpc_vertex * vertice_base,int *num_overlay,long int * prefix_overlay, gpc_vertex * vertice_overlay,int num,gpc_polygon * result,int *hole_base,int *hole_overlay,bounding_box *zOrder){
	int thid=blockIdx.x*blockDim.x+threadIdx.x;
	gpc_polygon base;
	gpc_vertex_list list_base;
	gpc_polygon overlay;
	gpc_vertex_list list_overlay;
	int pos1,pos2;
	//int pos11;
	pos1=zOrder[output[thid].nodeid-offset].index;
	pos2=output[thid].queryid;
	//pos1=384;
	//pos2=78;
	if(thid<num){
		//if(thid==116)
		//printf("(%d,%d,%d)\n",thid,pos1,pos2);
		list_base.num_vertices=num_base[pos1];
		//if(list_base.num_vertices==0){
			//printf("(%d,%d)\n",pos1,pos2);
		//}
		list_base.vertex=&(vertice_base[prefix_base[pos1]]);
		base.num_contours=1;
		base.hole=&(hole_base[pos1]);
		base.contour=&list_base;
		
		list_overlay.num_vertices=num_overlay[pos2];
		list_overlay.vertex=&(vertice_overlay[prefix_overlay[pos2]]);
		overlay.num_contours=1;
		overlay.hole=&(hole_overlay[pos2]);
		overlay.contour=&list_overlay;
		//printf("(%d,%d)\n",(output[thid].nodeid-offset),(output[thid].queryid));
		gpc_polygon_clip(GPC_INT, &base, &overlay,&(result[thid]));	
	}
}

__device__ void gpc_polygon_clip(gpc_op op, gpc_polygon *subj, gpc_polygon *clip, gpc_polygon *result)
{
  sb_tree       *sbtree= NULL;
  it_node       *it= NULL, *intersect;
  edge_node     *edge, *prev_edge, *next_edge, *succ_edge, *e0, *e1;
  edge_node     *aet= NULL, *c_heap= NULL, *s_heap= NULL;
  lmt_node      *lmt= NULL, *local_min;
  polygon_node  *out_poly= NULL, *p, *q, *poly, *npoly, *cf= NULL;
  vertex_node   *vtx, *nv;
  h_state        horiz[2];
  int            in[2], exists[2], parity[2]= {LEFT, LEFT};
  int            c, v, contributing, search, scanbeam= 0, sbt_entries= 0;
  int            vclass, bl, br, tl, tr;
  double        *sbt= NULL, xb, px, yb, yt, dy, ix, iy;

  /* Test for trivial NULL result cases */
  if (((subj->num_contours == 0) && (clip->num_contours == 0))
   || ((subj->num_contours == 0) && ((op == GPC_INT) || (op == GPC_DIFF)))
   || ((clip->num_contours == 0) &&  (op == GPC_INT)))
  {
    result->num_contours= 0;
    result->hole= NULL;
    result->contour= NULL;
    return;
  }

  /* Identify potentialy contributing contours */
  if (((op == GPC_INT) || (op == GPC_DIFF))
   && (subj->num_contours > 0) && (clip->num_contours > 0))
    minimax_test(subj, clip, op);

  /* Build LMT */
  if (subj->num_contours > 0)
    s_heap= build_lmt(&lmt, &sbtree, &sbt_entries, subj, SUBJ, op);
  if (clip->num_contours > 0)
    c_heap= build_lmt(&lmt, &sbtree, &sbt_entries, clip, CLIP, op);

  /* Return a NULL result if no contours contribute */
  if (lmt == NULL)
  {
    result->num_contours= 0;
    result->hole= NULL;
    result->contour= NULL;
    reset_lmt(&lmt);
    FREE(s_heap);
    FREE(c_heap);
    return;
  }

  /* Build scanbeam table from scanbeam tree */
  MALLOC(sbt, sbt_entries * sizeof(double), "sbt creation", double);
  build_sbt(&scanbeam, sbt, sbtree);
  scanbeam= 0;
  free_sbtree(&sbtree);

  /* Allow pointer re-use without causing memory leak */
  if (subj == result)
    gpc_free_polygon(subj);
  if (clip == result)
    gpc_free_polygon(clip);

  /* Invert clip polygon for difference operation */
  if (op == GPC_DIFF)
    parity[CLIP]= RIGHT;

  local_min= lmt;

  /* Process each scanbeam */
  while (scanbeam < sbt_entries)
  {
    /* Set yb and yt to the bottom and top of the scanbeam */
    yb= sbt[scanbeam++];
    if (scanbeam < sbt_entries)
    {
      yt= sbt[scanbeam];
      dy= yt - yb;
    }

    /* === SCANBEAM BOUNDARY PROCESSING ================================ */

    /* If LMT node corresponding to yb exists */
    if (local_min)
    {
      if (local_min->y == yb)
      {
        /* Add edges starting at this local minimum to the AET */
        for (edge= local_min->first_bound; edge; edge= edge->next_bound)
          add_edge_to_aet(&aet, edge, NULL);

        local_min= local_min->next;
      }
    }

    /* Set dummy previous x value */
    px= -DBL_MAX;

    /* Create bundles within AET */
    e0= aet;
    e1= aet;

    /* Set up bundle fields of first edge */
    aet->bundle[ABOVE][ aet->type]= (aet->top.y != yb);
    aet->bundle[ABOVE][!aet->type]= FALSE;
    aet->bstate[ABOVE]= UNBUNDLED;

    for (next_edge= aet->next; next_edge; next_edge= next_edge->next)
    {
      /* Set up bundle fields of next edge */
      next_edge->bundle[ABOVE][ next_edge->type]= (next_edge->top.y != yb);
      next_edge->bundle[ABOVE][!next_edge->type]= FALSE;
      next_edge->bstate[ABOVE]= UNBUNDLED;

      /* Bundle edges above the scanbeam boundary if they coincide */
      if (next_edge->bundle[ABOVE][next_edge->type])
      {
        if (EQ(e0->xb, next_edge->xb) && EQ(e0->dx, next_edge->dx)
	 && (e0->top.y != yb))
        {
          next_edge->bundle[ABOVE][ next_edge->type]^= 
            e0->bundle[ABOVE][ next_edge->type];
          next_edge->bundle[ABOVE][!next_edge->type]= 
            e0->bundle[ABOVE][!next_edge->type];
          next_edge->bstate[ABOVE]= BUNDLE_HEAD;
          e0->bundle[ABOVE][CLIP]= FALSE;
          e0->bundle[ABOVE][SUBJ]= FALSE;
          e0->bstate[ABOVE]= BUNDLE_TAIL;
        }
        e0= next_edge;
      }
    }
    
    horiz[CLIP]= NH;
    horiz[SUBJ]= NH;

    /* Process each edge at this scanbeam boundary */
    for (edge= aet; edge; edge= edge->next)
    {
      exists[CLIP]= edge->bundle[ABOVE][CLIP] + 
                   (edge->bundle[BELOW][CLIP] << 1);
      exists[SUBJ]= edge->bundle[ABOVE][SUBJ] + 
                   (edge->bundle[BELOW][SUBJ] << 1);

      if (exists[CLIP] || exists[SUBJ])
      {
        /* Set bundle side */
        edge->bside[CLIP]= parity[CLIP];
        edge->bside[SUBJ]= parity[SUBJ];

        /* Determine contributing status and quadrant occupancies */
        switch (op)
        {
        case GPC_DIFF:
        case GPC_INT:
          contributing= (exists[CLIP] && (parity[SUBJ] || horiz[SUBJ]))
                     || (exists[SUBJ] && (parity[CLIP] || horiz[CLIP]))
                     || (exists[CLIP] && exists[SUBJ]
                     && (parity[CLIP] == parity[SUBJ]));
          br= (parity[CLIP])
           && (parity[SUBJ]);
          bl= (parity[CLIP] ^ edge->bundle[ABOVE][CLIP])
           && (parity[SUBJ] ^ edge->bundle[ABOVE][SUBJ]);
          tr= (parity[CLIP] ^ (horiz[CLIP]!=NH))
           && (parity[SUBJ] ^ (horiz[SUBJ]!=NH));
          tl= (parity[CLIP] ^ (horiz[CLIP]!=NH) ^ edge->bundle[BELOW][CLIP]) 
           && (parity[SUBJ] ^ (horiz[SUBJ]!=NH) ^ edge->bundle[BELOW][SUBJ]);
          break;
        case GPC_XOR:
          contributing= exists[CLIP] || exists[SUBJ];
          br= (parity[CLIP])
            ^ (parity[SUBJ]);
          bl= (parity[CLIP] ^ edge->bundle[ABOVE][CLIP])
            ^ (parity[SUBJ] ^ edge->bundle[ABOVE][SUBJ]);
          tr= (parity[CLIP] ^ (horiz[CLIP]!=NH))
            ^ (parity[SUBJ] ^ (horiz[SUBJ]!=NH));
          tl= (parity[CLIP] ^ (horiz[CLIP]!=NH) ^ edge->bundle[BELOW][CLIP]) 
            ^ (parity[SUBJ] ^ (horiz[SUBJ]!=NH) ^ edge->bundle[BELOW][SUBJ]);
          break;
        case GPC_UNION:
          contributing= (exists[CLIP] && (!parity[SUBJ] || horiz[SUBJ]))
                     || (exists[SUBJ] && (!parity[CLIP] || horiz[CLIP]))
                     || (exists[CLIP] && exists[SUBJ]
                     && (parity[CLIP] == parity[SUBJ]));
          br= (parity[CLIP])
           || (parity[SUBJ]);
          bl= (parity[CLIP] ^ edge->bundle[ABOVE][CLIP])
           || (parity[SUBJ] ^ edge->bundle[ABOVE][SUBJ]);
          tr= (parity[CLIP] ^ (horiz[CLIP]!=NH))
           || (parity[SUBJ] ^ (horiz[SUBJ]!=NH));
          tl= (parity[CLIP] ^ (horiz[CLIP]!=NH) ^ edge->bundle[BELOW][CLIP]) 
           || (parity[SUBJ] ^ (horiz[SUBJ]!=NH) ^ edge->bundle[BELOW][SUBJ]);
          break;
        }

        /* Update parity */
        parity[CLIP]^= edge->bundle[ABOVE][CLIP];
        parity[SUBJ]^= edge->bundle[ABOVE][SUBJ];

        /* Update horizontal state */
        if (exists[CLIP])         
          horiz[CLIP]=
            next_h_state[horiz[CLIP]]
                        [((exists[CLIP] - 1) << 1) + parity[CLIP]];
        if (exists[SUBJ])         
          horiz[SUBJ]=
            next_h_state[horiz[SUBJ]]
                        [((exists[SUBJ] - 1) << 1) + parity[SUBJ]];

        vclass= tr + (tl << 1) + (br << 2) + (bl << 3);

        if (contributing)
        {
          xb= edge->xb;

          switch (vclass)
          {
          case EMN:
          case IMN:
            add_local_min(&out_poly, edge, xb, yb);
            px= xb;
            cf= edge->outp[ABOVE];
            break;
          case ERI:
            if (xb != px)
            {
              add_right(cf, xb, yb);
              px= xb;
            }
            edge->outp[ABOVE]= cf;
            cf= NULL;
            break;
          case ELI:
            add_left(edge->outp[BELOW], xb, yb);
            px= xb;
            cf= edge->outp[BELOW];
            break;
          case EMX:
            if (xb != px)
            {
              add_left(cf, xb, yb);
              px= xb;
            }
            merge_right(cf, edge->outp[BELOW], out_poly);
            cf= NULL;
            break;
          case ILI:
            if (xb != px)
            {
              add_left(cf, xb, yb);
              px= xb;
            }
            edge->outp[ABOVE]= cf;
            cf= NULL;
            break;
          case IRI:
            add_right(edge->outp[BELOW], xb, yb);
            px= xb;
            cf= edge->outp[BELOW];
            edge->outp[BELOW]= NULL;
            break;
          case IMX:
            if (xb != px)
            {
              add_right(cf, xb, yb);
              px= xb;
            }
            merge_left(cf, edge->outp[BELOW], out_poly);
            cf= NULL;
            edge->outp[BELOW]= NULL;
            break;
          case IMM:
            if (xb != px)
	    {
              add_right(cf, xb, yb);
              px= xb;
	    }
            merge_left(cf, edge->outp[BELOW], out_poly);
            edge->outp[BELOW]= NULL;
            add_local_min(&out_poly, edge, xb, yb);
            cf= edge->outp[ABOVE];
            break;
          case EMM:
            if (xb != px)
	    {
              add_left(cf, xb, yb);
              px= xb;
	    }
            merge_right(cf, edge->outp[BELOW], out_poly);
            edge->outp[BELOW]= NULL;
            add_local_min(&out_poly, edge, xb, yb);
            cf= edge->outp[ABOVE];
            break;
          case LED:
            if (edge->bot.y == yb)
              add_left(edge->outp[BELOW], xb, yb);
            edge->outp[ABOVE]= edge->outp[BELOW];
            px= xb;
            break;
          case RED:
            if (edge->bot.y == yb)
              add_right(edge->outp[BELOW], xb, yb);
            edge->outp[ABOVE]= edge->outp[BELOW];
            px= xb;
            break;
          default:
            break;
          } /* End of switch */
        } /* End of contributing conditional */
      } /* End of edge exists conditional */
    } /* End of AET loop */

    /* Delete terminating edges from the AET, otherwise compute xt */
    for (edge= aet; edge; edge= edge->next)
    {
      if (edge->top.y == yb)
      {
        prev_edge= edge->prev;
        next_edge= edge->next;
        if (prev_edge)
          prev_edge->next= next_edge;
        else
          aet= next_edge;
        if (next_edge)
          next_edge->prev= prev_edge;

        /* Copy bundle head state to the adjacent tail edge if required */
        if ((edge->bstate[BELOW] == BUNDLE_HEAD) && prev_edge)
	{
          if (prev_edge->bstate[BELOW] == BUNDLE_TAIL)
          {
            prev_edge->outp[BELOW]= edge->outp[BELOW];
            prev_edge->bstate[BELOW]= UNBUNDLED;
            if (prev_edge->prev)
              if (prev_edge->prev->bstate[BELOW] == BUNDLE_TAIL)
                prev_edge->bstate[BELOW]= BUNDLE_HEAD;
	  }
	}
      }
      else
      {
        if (edge->top.y == yt)
          edge->xt= edge->top.x;
        else
          edge->xt= edge->bot.x + edge->dx * (yt - edge->bot.y);
      }
    }

    if (scanbeam < sbt_entries)
    {
      /* === SCANBEAM INTERIOR PROCESSING ============================== */

      build_intersection_table(&it, aet, dy);

      /* Process each node in the intersection table */
      for (intersect= it; intersect; intersect= intersect->next)
      {
        e0= intersect->ie[0];
        e1= intersect->ie[1];

        /* Only generate output for contributing intersections */
        if ((e0->bundle[ABOVE][CLIP] || e0->bundle[ABOVE][SUBJ])
         && (e1->bundle[ABOVE][CLIP] || e1->bundle[ABOVE][SUBJ]))
	{
          p= e0->outp[ABOVE];
          q= e1->outp[ABOVE];
          ix= intersect->point.x;
          iy= intersect->point.y + yb;
 
          in[CLIP]= ( e0->bundle[ABOVE][CLIP] && !e0->bside[CLIP])
                 || ( e1->bundle[ABOVE][CLIP] &&  e1->bside[CLIP])
                 || (!e0->bundle[ABOVE][CLIP] && !e1->bundle[ABOVE][CLIP]
                     && e0->bside[CLIP] && e1->bside[CLIP]);
          in[SUBJ]= ( e0->bundle[ABOVE][SUBJ] && !e0->bside[SUBJ])
                 || ( e1->bundle[ABOVE][SUBJ] &&  e1->bside[SUBJ])
                 || (!e0->bundle[ABOVE][SUBJ] && !e1->bundle[ABOVE][SUBJ]
                     && e0->bside[SUBJ] && e1->bside[SUBJ]);
       
          /* Determine quadrant occupancies */
          switch (op)
          {
          case GPC_DIFF:
          case GPC_INT:
            tr= (in[CLIP])
             && (in[SUBJ]);
            tl= (in[CLIP] ^ e1->bundle[ABOVE][CLIP])
             && (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ]);
            br= (in[CLIP] ^ e0->bundle[ABOVE][CLIP])
             && (in[SUBJ] ^ e0->bundle[ABOVE][SUBJ]);
            bl= (in[CLIP] ^ e1->bundle[ABOVE][CLIP] ^ e0->bundle[ABOVE][CLIP])
             && (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ] ^ e0->bundle[ABOVE][SUBJ]);
            break;
          case GPC_XOR:
            tr= (in[CLIP])
              ^ (in[SUBJ]);
            tl= (in[CLIP] ^ e1->bundle[ABOVE][CLIP])
              ^ (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ]);
            br= (in[CLIP] ^ e0->bundle[ABOVE][CLIP])
              ^ (in[SUBJ] ^ e0->bundle[ABOVE][SUBJ]);
            bl= (in[CLIP] ^ e1->bundle[ABOVE][CLIP] ^ e0->bundle[ABOVE][CLIP])
              ^ (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ] ^ e0->bundle[ABOVE][SUBJ]);
            break;
          case GPC_UNION:
            tr= (in[CLIP])
             || (in[SUBJ]);
            tl= (in[CLIP] ^ e1->bundle[ABOVE][CLIP])
             || (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ]);
            br= (in[CLIP] ^ e0->bundle[ABOVE][CLIP])
             || (in[SUBJ] ^ e0->bundle[ABOVE][SUBJ]);
            bl= (in[CLIP] ^ e1->bundle[ABOVE][CLIP] ^ e0->bundle[ABOVE][CLIP])
             || (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ] ^ e0->bundle[ABOVE][SUBJ]);
            break;
          }
	  
          vclass= tr + (tl << 1) + (br << 2) + (bl << 3);

          switch (vclass)
          {
          case EMN:
            add_local_min(&out_poly, e0, ix, iy);
            e1->outp[ABOVE]= e0->outp[ABOVE];
            break;
          case ERI:
            if (p)
            {
              add_right(p, ix, iy);
              e1->outp[ABOVE]= p;
              e0->outp[ABOVE]= NULL;
            }
            break;
          case ELI:
            if (q)
            {
              add_left(q, ix, iy);
              e0->outp[ABOVE]= q;
              e1->outp[ABOVE]= NULL;
            }
            break;
          case EMX:
            if (p && q)
            {
              add_left(p, ix, iy);
              merge_right(p, q, out_poly);
              e0->outp[ABOVE]= NULL;
              e1->outp[ABOVE]= NULL;
            }
            break;
          case IMN:
            add_local_min(&out_poly, e0, ix, iy);
            e1->outp[ABOVE]= e0->outp[ABOVE];
            break;
          case ILI:
            if (p)
            {
              add_left(p, ix, iy);
              e1->outp[ABOVE]= p;
              e0->outp[ABOVE]= NULL;
            }
            break;
          case IRI:
            if (q)
            {
              add_right(q, ix, iy);
              e0->outp[ABOVE]= q;
              e1->outp[ABOVE]= NULL;
            }
            break;
          case IMX:
            if (p && q)
            {
              add_right(p, ix, iy);
              merge_left(p, q, out_poly);
              e0->outp[ABOVE]= NULL;
              e1->outp[ABOVE]= NULL;
            }
            break;
          case IMM:
            if (p && q)
            {
              add_right(p, ix, iy);
              merge_left(p, q, out_poly);
              add_local_min(&out_poly, e0, ix, iy);
              e1->outp[ABOVE]= e0->outp[ABOVE];
            }
            break;
          case EMM:
            if (p && q)
            {
              add_left(p, ix, iy);
              merge_right(p, q, out_poly);
              add_local_min(&out_poly, e0, ix, iy);
              e1->outp[ABOVE]= e0->outp[ABOVE];
            }
            break;
          default:
            break;
          } /* End of switch */
	} /* End of contributing intersection conditional */

        /* Swap bundle sides in response to edge crossing */
        if (e0->bundle[ABOVE][CLIP])
	  e1->bside[CLIP]= !e1->bside[CLIP];
        if (e1->bundle[ABOVE][CLIP])
	  e0->bside[CLIP]= !e0->bside[CLIP];
        if (e0->bundle[ABOVE][SUBJ])
	  e1->bside[SUBJ]= !e1->bside[SUBJ];
        if (e1->bundle[ABOVE][SUBJ])
	  e0->bside[SUBJ]= !e0->bside[SUBJ];

        /* Swap e0 and e1 bundles in the AET */
        prev_edge= e0->prev;
        next_edge= e1->next;
        if (next_edge)
          next_edge->prev= e0;

        if (e0->bstate[ABOVE] == BUNDLE_HEAD)
        {
          search= TRUE;
          while (search)
          {
            prev_edge= prev_edge->prev;
            if (prev_edge)
            {
              if (prev_edge->bstate[ABOVE] != BUNDLE_TAIL)
                search= FALSE;
            }
            else
              search= FALSE;
          }
        }
        if (!prev_edge)
        {
          aet->prev= e1;
          e1->next= aet;
          aet= e0->next;
        }
        else
        {
          prev_edge->next->prev= e1;
          e1->next= prev_edge->next;
          prev_edge->next= e0->next;
        }
        e0->next->prev= prev_edge;
        e1->next->prev= e1;
        e0->next= next_edge;
      } /* End of IT loop*/

      /* Prepare for next scanbeam */
      for (edge= aet; edge; edge= next_edge)
      {
        next_edge= edge->next;
        succ_edge= edge->succ;

        if ((edge->top.y == yt) && succ_edge)
        {
          /* Replace AET edge by its successor */
          succ_edge->outp[BELOW]= edge->outp[ABOVE];
          succ_edge->bstate[BELOW]= edge->bstate[ABOVE];
          succ_edge->bundle[BELOW][CLIP]= edge->bundle[ABOVE][CLIP];
          succ_edge->bundle[BELOW][SUBJ]= edge->bundle[ABOVE][SUBJ];
          prev_edge= edge->prev;
          if (prev_edge)
            prev_edge->next= succ_edge;
          else
            aet= succ_edge;
          if (next_edge)
            next_edge->prev= succ_edge;
          succ_edge->prev= prev_edge;
          succ_edge->next= next_edge;
        }
        else
        {
          /* Update this edge */
          edge->outp[BELOW]= edge->outp[ABOVE];
          edge->bstate[BELOW]= edge->bstate[ABOVE];
          edge->bundle[BELOW][CLIP]= edge->bundle[ABOVE][CLIP];
          edge->bundle[BELOW][SUBJ]= edge->bundle[ABOVE][SUBJ];
          edge->xb= edge->xt;
	      }
        edge->outp[ABOVE]= NULL;
      }
    }
  } /* === END OF SCANBEAM PROCESSING ================================== */

  /* Generate result polygon from out_poly */
  result->contour= NULL;
  result->hole= NULL;
  result->num_contours= count_contours(out_poly);
  if (result->num_contours > 0)
  {
    MALLOC(result->hole, result->num_contours
           * sizeof(int), "hole flag table creation", int);
    MALLOC(result->contour, result->num_contours
           * sizeof(gpc_vertex_list), "contour creation", gpc_vertex_list);

    c= 0;
    for (poly= out_poly; poly; poly= npoly)
    {
      npoly= poly->next;
      if (poly->active)
      {
        result->hole[c]= poly->proxy->hole;
        result->contour[c].num_vertices= poly->active;
        MALLOC(result->contour[c].vertex,
          result->contour[c].num_vertices * sizeof(gpc_vertex),
          "vertex creation", gpc_vertex);
      
        v= result->contour[c].num_vertices - 1;
        for (vtx= poly->proxy->v[LEFT]; vtx; vtx= nv)
        {
          nv= vtx->next;
          result->contour[c].vertex[v].x= vtx->x;
          result->contour[c].vertex[v].y= vtx->y;
          FREE(vtx);
          v--;
        }
        c++;
      }
      FREE(poly);
    }
  }
  else
  {
    for (poly= out_poly; poly; poly= npoly)
    {
      npoly= poly->next;
      FREE(poly);
    }
  }


  /* Tidy up */
  reset_it(&it);
  reset_lmt(&lmt);
  FREE(c_heap);
  FREE(s_heap);
  FREE(sbt);
}

void readData(char * base_filename,char * overlay_filename, int **rect1,int **rect2,int **rect3,int **rect4,int * num_contours,int **rect1_query,int **rect2_query,int **rect3_query,int **rect4_query,int *numOfQuerys,int ** num_base,long int **prefix_base,gpc_vertex ** vertex_base,int **num_overlay,long int **prefix_overlay,gpc_vertex **vertex_overlay,int *minX,int *minY,int **hole_base_cpu,int **hole_overlay_cpu){
	
	FILE *basefp,*overlayfp;
	gpc_vertex_list *contour_base,*contour_overlay;
	int * hole_base,*hole_overlay;
	int * rect1_temp,*rect2_temp,*rect3_temp,*rect4_temp;
	int * rect1_query_temp,*rect2_query_temp,*rect3_query_temp,*rect4_query_temp;
	int num_contours_temp, numOfQuerys_temp;
	int *num_base_temp,*num_overlay_temp;
	long int *prefix_base_temp, *prefix_overlay_temp;
	gpc_vertex * vertex_base_temp, *vertex_overlay_temp;
	basefp=fopen(base_filename,"r");
	overlayfp=fopen(overlay_filename,"r");
	int i,j;
  	fscanf(basefp, "%d", &num_contours_temp);
  	fscanf(overlayfp, "%d", &numOfQuerys_temp);
  	double temp1,temp2,temp3,temp4;
  	gpc_vertex *vertex_base_cpu,*vertex_overlay_cpu;

  	MALLOC_CPU(num_base_temp, num_contours_temp * sizeof(int),"num base array creation", int);
  	MALLOC_CPU(hole_base, num_contours_temp * sizeof(int),"hole base array creation", int);
  	MALLOC_CPU(prefix_base_temp, num_contours_temp * sizeof(long int),"prefix base array creation", long int);
  	MALLOC_CPU(contour_base, num_contours_temp * sizeof(gpc_vertex_list), "base contour creation", gpc_vertex_list);
  	
  	rect1_temp=(int *)malloc(num_contours_temp*sizeof(int));
    rect2_temp=(int *)malloc(num_contours_temp*sizeof(int));
    rect3_temp=(int *)malloc(num_contours_temp*sizeof(int));
    rect4_temp=(int *)malloc(num_contours_temp*sizeof(int));
  	
  	MALLOC_CPU(rect1_query_temp, numOfQuerys_temp * sizeof(int),"rect1_query creation", int);
	MALLOC_CPU(rect2_query_temp, numOfQuerys_temp * sizeof(int),"rect2_query creation", int);
	MALLOC_CPU(rect3_query_temp, numOfQuerys_temp* sizeof(int),"rect3_query creation", int);
	MALLOC_CPU(rect4_query_temp, numOfQuerys_temp* sizeof(int),"rect4_query creation", int);
  	
  	
  	MALLOC_CPU(num_overlay_temp, numOfQuerys_temp * sizeof(int),"num overlay array creation", int);
  	MALLOC_CPU(hole_overlay, numOfQuerys_temp * sizeof(int),"hole overlay array creation", int);
  	MALLOC_CPU(prefix_overlay_temp, numOfQuerys_temp * sizeof(long int),"prefix overlay array creation", long int);
  	MALLOC_CPU(contour_overlay, numOfQuerys_temp* sizeof(gpc_vertex_list), "overlay contour creation", gpc_vertex_list);
  	
	for (i=0;i<num_contours_temp;i++)
  	{
    	fscanf(basefp, "%d", &(num_base_temp[i]));
      	fscanf(basefp, "%d", &(hole_base[i]));
		if(i==0){
			prefix_base_temp[i]=0;
		}
		else{
			prefix_base_temp[i]=prefix_base_temp[i-1]+num_base_temp[i-1];
		}
    	MALLOC_CPU(contour_base[i].vertex, num_base_temp[i] * sizeof(gpc_vertex), "vertex creation input", gpc_vertex);
    	fscanf(basefp, "%lf %lf", &(temp1),&(temp2));
    	fscanf(basefp, "%lf %lf", &(temp3),&(temp4));
    	rect1_temp[i]=(int)(temp1*100000);
    	rect2_temp[i]=(int)(temp2*100000);
    	rect3_temp[i]=(int)(temp3*100000);
    	rect4_temp[i]=(int)(temp4*100000);
    	if(rect1_temp[i]<(*minX)){
			(*minX)=rect1_temp[i];
		}
		if(rect2_temp[i]<(*minY)){
			(*minY)=rect2_temp[i];
		}
    	for (j= 0; j < num_base_temp[i]; j++){
    	  	fscanf(basefp, "%lf %lf", &(contour_base[i].vertex[j].x),&(contour_base[i].vertex[j].y));
  		}
  	}
	MALLOC_CPU(vertex_base_cpu, (num_base_temp[num_contours_temp-1]+prefix_base_temp[num_contours_temp-1]) * sizeof(gpc_vertex),"vertex array creation", gpc_vertex);
	(cudaMalloc( (void**) &vertex_base_temp, sizeof(gpc_vertex)*(num_base_temp[num_contours_temp-1]+prefix_base_temp[num_contours_temp-1])));

	for(i=0;i<num_contours_temp;i++){
		for(j=0;j<(num_base_temp[i]);j++){
			vertex_base_cpu[prefix_base_temp[i]+j]=contour_base[i].vertex[j];	
		}
		FREE_CPU(contour_base[i].vertex);
	}
	(cudaMemcpy( vertex_base_temp, vertex_base_cpu, sizeof(gpc_vertex)*(num_base_temp[num_contours_temp-1]+prefix_base_temp[num_contours_temp-1]), cudaMemcpyHostToDevice));
	FREE_CPU(vertex_base_cpu);

	
	for (i=0;i<numOfQuerys_temp;i++)
  	{
    	fscanf(overlayfp, "%d", &(num_overlay_temp[i]));
      	fscanf(overlayfp, "%d", &(hole_overlay[i]));
		if(i==0){
			prefix_overlay_temp[i]=0;
		}
		else{
			prefix_overlay_temp[i]=prefix_overlay_temp[i-1]+num_overlay_temp[i-1];
		}
    	MALLOC_CPU(contour_overlay[i].vertex, num_overlay_temp[i] * sizeof(gpc_vertex), "vertex creation input", gpc_vertex);
    	fscanf(overlayfp, "%lf %lf", &(temp1),&(temp2));
    	fscanf(overlayfp, "%lf %lf", &(temp3),&(temp4));
    	rect1_query_temp[i]=(int)(temp1*100000);
    	rect2_query_temp[i]=(int)(temp2*100000);
    	rect3_query_temp[i]=(int)(temp3*100000);
    	rect4_query_temp[i]=(int)(temp4*100000);
    	for (j= 0; j < num_overlay_temp[i]; j++){
    	  	fscanf(overlayfp, "%lf %lf", &(contour_overlay[i].vertex[j].x),&(contour_overlay[i].vertex[j].y));
  		}
  	}
	MALLOC_CPU(vertex_overlay_cpu, (num_overlay_temp[numOfQuerys_temp-1]+prefix_overlay_temp[numOfQuerys_temp-1]) * sizeof(gpc_vertex),"vertex array creation", gpc_vertex);	
	(cudaMalloc( (void**) &vertex_overlay_temp, sizeof(gpc_vertex)* (num_overlay_temp[numOfQuerys_temp-1]+prefix_overlay_temp[numOfQuerys_temp-1])));

	for(i=0;i<numOfQuerys_temp;i++){
		for(j=0;j<num_overlay_temp[i];j++){
			vertex_overlay_cpu[prefix_overlay_temp[i]+j]=contour_overlay[i].vertex[j];	
		}
		FREE_CPU(contour_overlay[i].vertex);
	}
	(cudaMemcpy( vertex_overlay_temp, vertex_overlay_cpu, sizeof(gpc_vertex)*(num_overlay_temp[numOfQuerys_temp-1]+prefix_overlay_temp[numOfQuerys_temp-1]), cudaMemcpyHostToDevice));
	FREE_CPU(vertex_overlay_cpu);
	
	//(*numOfVertices_base)=((*num_base)[(*num_contours)-1]+(*prefix_base)[(*num_contours)-1]);
	//(*numOfVertices_overlay)=((*num_overlay)[(*numOfQuerys)-1]+(*prefix_overlay)[(*numOfQuerys)-1]);
	
	(*rect1)=rect1_temp;
	(*rect2)=rect2_temp;
	(*rect3)=rect3_temp;
	(*rect4)=rect4_temp;
	(*rect1_query)=rect1_query_temp;
	(*rect2_query)=rect2_query_temp;
	(*rect3_query)=rect3_query_temp;
	(*rect4_query)=rect4_query_temp;
	(*num_contours)=num_contours_temp;
	(*numOfQuerys)=numOfQuerys_temp;
	(*num_base)=num_base_temp;
	(*num_overlay)=num_overlay_temp;
	(*prefix_base)=prefix_base_temp;
	(*prefix_overlay)=prefix_overlay_temp;
	(*vertex_base)=vertex_base_temp;
	(*vertex_overlay)=vertex_overlay_temp;
	(*hole_base_cpu)=hole_base;
	(*hole_overlay_cpu)=hole_overlay;
	fclose(basefp);
	fclose(overlayfp);
}
	
__device__ void gpc_free_polygon(gpc_polygon *p)
{

  int c;

  for (c= 0; c < p->num_contours; c++)
  FREE(p->contour[c].vertex);
  FREE(p->hole);
  FREE(p->contour);
  p->num_contours= 0;

}

__device__ void reset_it(it_node **it)
{
  it_node *itn;

  while (*it)
  {
    itn= (*it)->next;
    FREE(*it);
    *it= itn;
  }
}

__device__  void reset_lmt(lmt_node **lmt)
{
  lmt_node *lmtn;

  while (*lmt)
  {


    lmtn= (*lmt)->next;
    FREE(*lmt);
    *lmt= lmtn;
  }
}

__device__  void insert_bound(edge_node **b, edge_node *e)
{

  edge_node *existing_bound;

L1:if (!*b)
  {
    /* Link node e to the tail of the list */
    *b= e;
  }
  else
  {
    /* Do primary sort on the x field */
    if (e[0].bot.x < (*b)[0].bot.x)
    {
      /* Insert a new node mid-list */
      existing_bound= *b;
      *b= e;
      (*b)->next_bound= existing_bound;
    }
    else
    {
      if (e[0].bot.x == (*b)[0].bot.x)
      {
        /* Do secondary sort on the dx field */
        if (e[0].dx < (*b)[0].dx)
        {
          /* Insert a new node mid-list */
          existing_bound= *b;
          *b= e;
          (*b)->next_bound= existing_bound;
        }
        else
        {
          /* Head further down the list */
         // insert_bound(&((*b)->next_bound), e);
         b=&((*b)->next_bound);
         goto L1;
        }
      }
      else
      {
        /* Head further down the list */
        //insert_bound(&((*b)->next_bound), e);
        b=&((*b)->next_bound);
         goto L1;
      }
    }
  }
}


__device__  edge_node **bound_list(lmt_node **lmt, double y)
{
  lmt_node *existing_node;

L1:if (!*lmt)
  {
    /* Add node onto the tail end of the LMT */
    MALLOC(*lmt, sizeof(lmt_node), "LMT insertion", lmt_node);
    (*lmt)->y= y;
    (*lmt)->first_bound= NULL;
    (*lmt)->next= NULL;
    return &((*lmt)->first_bound);
  }
  else
    if (y < (*lmt)->y)
    {
      /* Insert a new LMT node before the current node */
      existing_node= *lmt;
      MALLOC(*lmt, sizeof(lmt_node), "LMT insertion", lmt_node);
      (*lmt)->y= y;
      (*lmt)->first_bound= NULL;
      (*lmt)->next= existing_node;
      return &((*lmt)->first_bound);
    }
    else
      if (y > (*lmt)->y)
      { 
       // return bound_list(&((*lmt)->next), y);
        lmt=&((*lmt)->next);
        goto L1;
      }
      else
        /* Use this existing LMT node */
        return &((*lmt)->first_bound);
}



__device__  void add_to_sbtree(int *entries, sb_tree **sbtree, double y)
{
L1:if (!*sbtree)
  {
    /* Add a new tree node here */
    MALLOC(*sbtree, sizeof(sb_tree), "scanbeam tree insertion", sb_tree);
    (*sbtree)->y= y;
    (*sbtree)->visited=0;
    (*sbtree)->less= NULL;
    (*sbtree)->more= NULL;
    (*entries)++;
  }
  else
  {
    if ((*sbtree)->y > y)
    {
     //add_to_sbtree(entries, &((*sbtree)->less), y);
      sbtree=&((*sbtree)->less); 
      goto L1; 
    }
    else
    {
      if ((*sbtree)->y < y)
      {
        /* Head into the 'more' sub-tree */
       // add_to_sbtree(entries, &((*sbtree)->more), y);
       sbtree=&((*sbtree)->more); 
      goto L1;
      }
    }
  }
}


__device__ void build_sbt(int *entries,double *sbt,sb_tree  *sbtree) 
{
	sb_tree *current,*pre;
	if(sbtree == NULL)	return;
	current = sbtree;
	while(current != NULL)
	{
		if(current->less == NULL)
		{
			sbt[*entries]=current->y;
			(*entries)++;
			current = current->more;
		}
		else
		{
			pre = current->less;
			while(pre->more != NULL && pre->more != current)
			pre = pre->more;
			if(pre->more == NULL)
			{
				pre->more = current;
				current = current->less;
			}
			else
			{
				pre->more = NULL;
				sbt[*entries]=current->y;
				(*entries)++;
				current = current->more;
			}
		}
	}
}


__device__  void free_sbtree(sb_tree **sbtree)                              
{
	{
	/*
	sb_tree *stack[100];
	int top,sig, sign[100];
	top = -1;
	if(*sbtree != NULL)
	{
		top++;
		stack[top] = *sbtree;
		sign [top]=1;
		if((*sbtree)->more != NULL)
		{
			top++;
			stack[top] = (*sbtree)->more;
			sign [top]=-1;
		}
		sbtree = &((*sbtree)->less);
		while(top >= 0)
		{
			while ( *sbtree!= NULL)
			{
				top++;
				stack[top] = *sbtree;
				sign [top]=1;
				if((*sbtree)->more != NULL)
				{
					top++;
					stack[top] = (*sbtree)->more;
					sign[top]=-1;
				}
			sbtree =&((*sbtree)->less);
			}
			sbtree = &(stack[top]);
			sig=sign[top];
			top--;
			while((sig > 0) && (top >= -1))
			{
				FREE(*sbtree);
				sbtree = &(stack[top]);
				sig=sign[top];
				top--;
			}
		}
	}
	*/
	}
   sb_tree * nodeStack[400];
   sb_tree * currNode;
   int head=-1;
   nodeStack[++head]=(*sbtree);
   while (head!=-1){
       currNode = nodeStack[head];
       if ((currNode->less) &&(currNode->less->visited == 0)){
       		nodeStack[++head]=(currNode->less);
       }
       else if ((currNode->more) && (currNode->more->visited == 0)){
       	 	nodeStack[++head]=(currNode->more);
       }
       else{
       		currNode->visited=1;
       		if(currNode->less){
       			FREE(currNode->less);
       		}
       		if(currNode->more){
       			FREE(currNode->more);
       		}
       		head--;
       }
    }
}


__device__  int count_optimal_vertices(gpc_vertex_list c)
{
  int result= 0, i;

  if (c.num_vertices > 0)
  {
    for (i= 0; i < c.num_vertices; i++)
      if (OPTIMAL(c.vertex, i, c.num_vertices))
        result++;
  }
  return result;
}

__device__  edge_node *build_lmt(lmt_node **lmt, sb_tree **sbtree,
                            int *sbt_entries, gpc_polygon *p, int type,
                            gpc_op op)
{

  int          c, i, min, max, num_edges, v, num_vertices;
  int          total_vertices= 0, e_index=0;
  edge_node   *e, *edge_table;


  for (c= 0; c < p->num_contours; c++)
    total_vertices+= count_optimal_vertices(p->contour[c]);

  //printf("total vertices is %d\n",total_vertices);
  MALLOC(edge_table,total_vertices*sizeof(edge_node),"edge_node",edge_node);


  for (c= 0; c < p->num_contours; c++)
  {
    if (p->contour[c].num_vertices < 0)
    {
      p->contour[c].num_vertices= -p->contour[c].num_vertices;
    }
    else
    {
      num_vertices= 0;
      for (i= 0; i < p->contour[c].num_vertices; i++)
        if (OPTIMAL(p->contour[c].vertex, i, p->contour[c].num_vertices))
        {
          edge_table[num_vertices].vertex.x= p->contour[c].vertex[i].x;
          edge_table[num_vertices].vertex.y= p->contour[c].vertex[i].y;

          add_to_sbtree(sbt_entries, sbtree,
                        edge_table[num_vertices].vertex.y);

          num_vertices++;
        }

      for (min= 0; min < num_vertices; min++)
      {
        if (FWD_MIN(edge_table, min, num_vertices))
        {
          num_edges= 1;
          max= NEXT_INDEX(min, num_vertices);
          while (NOT_FMAX(edge_table, max, num_vertices))
          {
            num_edges++;
            max= NEXT_INDEX(max, num_vertices);
          }

          e= &edge_table[e_index];
          e_index+= num_edges;
          v= min;
          e[0].bstate[BELOW]= UNBUNDLED;
          e[0].bundle[BELOW][CLIP]= FALSE;
          e[0].bundle[BELOW][SUBJ]= FALSE;
          for (i= 0; i < num_edges; i++)
          {
            e[i].xb= edge_table[v].vertex.x;
            e[i].bot.x= edge_table[v].vertex.x;
            e[i].bot.y= edge_table[v].vertex.y;

            v= NEXT_INDEX(v, num_vertices);

            e[i].top.x= edge_table[v].vertex.x;
            e[i].top.y= edge_table[v].vertex.y;
            e[i].dx= (edge_table[v].vertex.x - e[i].bot.x) /
                       (e[i].top.y - e[i].bot.y);
            e[i].type= type;
            e[i].outp[ABOVE]= NULL;
            e[i].outp[BELOW]= NULL;
            e[i].next= NULL;
            e[i].prev= NULL;
            e[i].succ= ((num_edges > 1) && (i < (num_edges - 1))) ?
                       &(e[i + 1]) : NULL;
            e[i].pred= ((num_edges > 1) && (i > 0)) ? &(e[i - 1]) : NULL;
            e[i].next_bound= NULL;
            e[i].bside[CLIP]= (op == GPC_DIFF) ? RIGHT : LEFT;
            e[i].bside[SUBJ]= LEFT;
          }
          insert_bound(bound_list(lmt, edge_table[min].vertex.y), e);
        }
      }

      for (min= 0; min < num_vertices; min++)
      {
        if (REV_MIN(edge_table, min, num_vertices))
        {
          num_edges= 1;
          max= PREV_INDEX(min, num_vertices);
          while (NOT_RMAX(edge_table, max, num_vertices))
          {
            num_edges++;
            max= PREV_INDEX(max, num_vertices);
          }

          e= &edge_table[e_index];
          e_index+= num_edges;
          v= min;
          e[0].bstate[BELOW]= UNBUNDLED;
          e[0].bundle[BELOW][CLIP]= FALSE;
          e[0].bundle[BELOW][SUBJ]= FALSE;
          for (i= 0; i < num_edges; i++)
          {
            e[i].xb= edge_table[v].vertex.x;
            e[i].bot.x= edge_table[v].vertex.x;
            e[i].bot.y= edge_table[v].vertex.y;

            v= PREV_INDEX(v, num_vertices);

            e[i].top.x= edge_table[v].vertex.x;
            e[i].top.y= edge_table[v].vertex.y;
            e[i].dx= (edge_table[v].vertex.x - e[i].bot.x) /
                       (e[i].top.y - e[i].bot.y);
            e[i].type= type;
            e[i].outp[ABOVE]= NULL;
            e[i].outp[BELOW]= NULL;
            e[i].next= NULL;
            e[i].prev= NULL;
            e[i].succ= ((num_edges > 1) && (i < (num_edges - 1))) ?
                       &(e[i + 1]) : NULL;
            e[i].pred= ((num_edges > 1) && (i > 0)) ? &(e[i - 1]) : NULL;
            e[i].next_bound= NULL;
            e[i].bside[CLIP]= (op == GPC_DIFF) ? RIGHT : LEFT;
            e[i].bside[SUBJ]= LEFT;
          }
          insert_bound(bound_list(lmt, edge_table[min].vertex.y), e);
        }
      }
    }
  }
  return edge_table;
}

__device__  void add_edge_to_aet(edge_node **aet, edge_node *edge, edge_node *prev)  
{
L1:if (!*aet)
  {
    *aet= edge;
    edge->prev= prev;
    edge->next= NULL;
  }
  else
  {
    if (edge->xb < (*aet)->xb)
    {
      edge->prev= prev;
      edge->next= *aet;
      (*aet)->prev= edge;
      *aet= edge;
    }
    else
    {
      if (edge->xb == (*aet)->xb)
      {
        if (edge->dx < (*aet)->dx)
        {
          edge->prev= prev;
          edge->next= *aet;
          (*aet)->prev= edge;
          *aet= edge;
        }
        else
        {
          //add_edge_to_aet(&((*aet)->next), edge, *aet);
           prev=*aet;
           aet=&((*aet)->next);
           goto L1;
        }
      }
      else
      {
        //add_edge_to_aet(&((*aet)->next), edge, *aet);
         prev=*aet;
         aet=&((*aet)->next);
         goto L1;
      
       }
    }
  }
}

__device__  void add_intersection(it_node **it, edge_node *edge0, edge_node *edge1,
                             double x, double y)
{

  it_node *existing_node;

L1:if (!*it)
  {
    /* Append a new node to the tail of the list */
    MALLOC(*it, sizeof(it_node), "IT insertion", it_node);
    (*it)->ie[0]= edge0;
    (*it)->ie[1]= edge1;
    (*it)->point.x= x;
    (*it)->point.y= y;
    (*it)->next= NULL;
  }
  else
  {
    if ((*it)->point.y > y)
    {
      /* Insert a new node mid-list */
      existing_node= *it;
      MALLOC(*it, sizeof(it_node), "IT insertion", it_node);
      (*it)->ie[0]= edge0;
      (*it)->ie[1]= edge1;
      (*it)->point.x= x;
      (*it)->point.y= y;
      (*it)->next= existing_node;
    }
    else
     { /* Head further down the list */
     // add_intersection(&((*it)->next), edge0, edge1, x, y);
      it=&((*it)->next);
      goto L1;
     }
  }
}

__device__  void add_st_edge(st_node **st, it_node **it, edge_node *edge,
                        double dy)
{
  st_node *existing_node;
  double   den, r, x, y;

L1:if (!*st)
  {
    /* Append edge onto the tail end of the ST */
    MALLOC(*st, sizeof(st_node), "ST insertion", st_node);
    (*st)->edge= edge;
    (*st)->xb= edge->xb;
    (*st)->xt= edge->xt;
    (*st)->dx= edge->dx;
    (*st)->prev= NULL;
  }
  else
  {
    den= ((*st)->xt - (*st)->xb) - (edge->xt - edge->xb);

    /* If new edge and ST edge don't cross */
    if ((edge->xt >= (*st)->xt) || (edge->dx == (*st)->dx) || 
        (fabs(den) <= DBL_EPSILON))
    {
      /* No intersection - insert edge here (before the ST edge) */
      existing_node= *st;
      MALLOC(*st, sizeof(st_node), "ST insertion", st_node);
      (*st)->edge= edge;
      (*st)->xb= edge->xb;
      (*st)->xt= edge->xt;
      (*st)->dx= edge->dx;
      (*st)->prev= existing_node;
    }
    else
    {
      /* Compute intersection between new edge and ST edge */
      r= (edge->xb - (*st)->xb) / den;
      x= (*st)->xb + r * ((*st)->xt - (*st)->xb);
      y= r * dy;

      /* Insert the edge pointers and the intersection point in the IT */
      add_intersection(it, (*st)->edge, edge, x, y);

      /* Head further into the ST */
     // add_st_edge(&((*st)->prev), it, edge, dy);

      st=&((*st)->prev);
      goto L1;
    }
  }
}

__device__  void build_intersection_table(it_node **it, edge_node *aet, double dy)
{
  st_node   *st, *stp;
  edge_node *edge;

  /* Build intersection table for the current scanbeam */
  reset_it(it);
  st= NULL;

  /* Process each AET edge */
  for (edge= aet; edge; edge= edge->next)
  {
    if ((edge->bstate[ABOVE] == BUNDLE_HEAD) ||
         edge->bundle[ABOVE][CLIP] || edge->bundle[ABOVE][SUBJ])
      add_st_edge(&st, it, edge, dy);
  }

  /* Free the sorted edge table */
  while (st)
  {
    stp= st->prev;
    FREE(st);
    st= stp;
  }
}

__device__  int count_contours(polygon_node *polygon)
{
  int          nc, nv;
  vertex_node *v, *nextv;

  for (nc= 0; polygon; polygon= polygon->next)
    if (polygon->active)
    {
      /* Count the vertices in the current contour */
      nv= 0;
      for (v= polygon->proxy->v[LEFT]; v; v= v->next)
        nv++;

      /* Record valid vertex counts in the active field */
      if (nv > 2)
      {
        polygon->active= nv;
        nc++;
      }
      else
      {
        /* Invalid contour: just free the heap */
        for (v= polygon->proxy->v[LEFT]; v; v= nextv)
        {
          nextv= v->next;
          FREE(v);
        }
        polygon->active= 0;
      }
    }
  return nc;
}

__device__  void add_left(polygon_node *p, double x, double y)
{
  vertex_node *nv;

  /* Create a new vertex node and set its fields */
  MALLOC(nv,sizeof(vertex_node),"vertex_node",vertex_node);
  nv->x= x;
  nv->y= y;

  /* Add vertex nv to the left end of the polygon's vertex list */


   nv->next= p->proxy->v[LEFT];

  /* Update proxy->[LEFT] to point to nv */
  p->proxy->v[LEFT]= nv;
}

__device__  void merge_left(polygon_node *p, polygon_node *q, polygon_node *list)
{
  polygon_node *target;

  /* Label contour as a hole */
  q->proxy->hole= TRUE;

  if (p->proxy != q->proxy)
  {
    /* Assign p's vertex list to the left end of q's list */
    p->proxy->v[RIGHT]->next= q->proxy->v[LEFT];
    q->proxy->v[LEFT]= p->proxy->v[LEFT];

    /* Redirect any p->proxy references to q->proxy */
    
    for (target= p->proxy; list; list= list->next)
    {
      if (list->proxy == target)
      {
        list->active= FALSE;
        list->proxy= q->proxy;
      }
    }
  }
}

__device__  void add_right(polygon_node *p, double x, double y)
{
  vertex_node *nv;

  /* Create a new vertex node and set its fields */
  MALLOC(nv,sizeof(vertex_node),"vertex_node",vertex_node);
  nv->x= x;
  nv->y= y;
  nv->next= NULL;

  /* Add vertex nv to the right end of the polygon's vertex list */
  p->proxy->v[RIGHT]->next= nv;

  /* Update proxy->v[RIGHT] to point to nv */
  p->proxy->v[RIGHT]= nv;
}

__device__  void merge_right(polygon_node *p, polygon_node *q, polygon_node *list)
{
  polygon_node *target;

  /* Label contour as external */
  q->proxy->hole= FALSE;

  if (p->proxy != q->proxy)
  {
    /* Assign p's vertex list to the right end of q's list */
    q->proxy->v[RIGHT]->next= p->proxy->v[LEFT];
    q->proxy->v[RIGHT]= p->proxy->v[RIGHT];

    /* Redirect any p->proxy references to q->proxy */
    for (target= p->proxy; list; list= list->next)
    {
      if (list->proxy == target)
      {
        list->active= FALSE;
        list->proxy= q->proxy;
      }
    }
  }
}

__device__  void add_local_min(polygon_node **p, edge_node *edge,
                          double x, double y)
{
  polygon_node *existing_min;
  vertex_node  *nv;

  existing_min= *p;

 

  MALLOC(*p,sizeof(polygon_node),"polygon_node",polygon_node);

  /* Create a new vertex node and set its fields */
  MALLOC(nv,sizeof(vertex_node),"vertex_node",vertex_node);
  nv->x= x;
  nv->y= y;
  nv->next= NULL;
  /* Initialise proxy to point to p itself */
  (*p)->proxy= (*p);
  (*p)->active= TRUE;
   
   (*p)->next= existing_min;

  /* Make v[LEFT] and v[RIGHT] point to new vertex nv */
  (*p)->v[LEFT]= nv;
  (*p)->v[RIGHT]= nv;

  /* Assign polygon p to the edge */
  edge->outp[ABOVE]= *p;
}

__device__   bbox *create_contour_bboxes(gpc_polygon *p)
{
  bbox *box;
  int   c, v;

  
 MALLOC(box,p->num_contours*sizeof(bbox),"bbox",bbox);

  /* Construct contour bounding boxes */
  for (c= 0; c < p->num_contours; c++)
  {
    /* Initialise bounding box extent */
    box[c].xmin= DBL_MAX;
    box[c].ymin= DBL_MAX;
    box[c].xmax= -DBL_MAX;
    box[c].ymax= -DBL_MAX;

    for (v= 0; v < p->contour[c].num_vertices; v++)
    {
      /* Adjust bounding box */
      if (p->contour[c].vertex[v].x < box[c].xmin)
        box[c].xmin= p->contour[c].vertex[v].x;
      if (p->contour[c].vertex[v].y < box[c].ymin)
        box[c].ymin= p->contour[c].vertex[v].y;
      if (p->contour[c].vertex[v].x > box[c].xmax)
        box[c].xmax= p->contour[c].vertex[v].x;
      if (p->contour[c].vertex[v].y > box[c].ymax)
          box[c].ymax= p->contour[c].vertex[v].y;
    }
  }
  return box;  
}

__device__  void minimax_test(gpc_polygon *subj, gpc_polygon *clip,gpc_op op)
{
  bbox *s_bbox, *c_bbox;
  int   s, c, *o_table, overlap;

  s_bbox= create_contour_bboxes(subj);
  c_bbox= create_contour_bboxes(clip);


  MALLOC( o_table,subj->num_contours * clip->num_contours*sizeof(int),"int",int);



  for (s= 0; s < subj->num_contours; s++)
    for (c= 0; c < clip->num_contours; c++)
      o_table[c * subj->num_contours + s]=
             (!((s_bbox[s].xmax < c_bbox[c].xmin) ||
                (s_bbox[s].xmin > c_bbox[c].xmax))) &&
             (!((s_bbox[s].ymax < c_bbox[c].ymin) ||
                (s_bbox[s].ymin > c_bbox[c].ymax)));


  for (c= 0; c < clip->num_contours; c++)
  {
    overlap= 0;
    for (s= 0; (!overlap) && (s < subj->num_contours); s++)
      overlap= o_table[c * subj->num_contours + s];

    if (!overlap)
      clip->contour[c].num_vertices = -clip->contour[c].num_vertices;
  
 
}  


  if (op == GPC_INT)
  {  
    for (s= 0; s < subj->num_contours; s++)
    {
      overlap= 0;
      for (c= 0; (!overlap) && (c < clip->num_contours); c++)
        overlap= o_table[c * subj->num_contours + s];

      if (!overlap)
        subj->contour[s].num_vertices = -subj->contour[s].num_vertices;
    }  
  }


  FREE(s_bbox);
  FREE(c_bbox);
  FREE(o_table);

}
