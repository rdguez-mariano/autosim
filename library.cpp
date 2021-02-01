#include "library.h"


void draw_line(double *igray, int a0, int b0, int a1, int b1, float value, int width, int height)
{

  int bdx,bdy;
  int sx,sy,dx,dy,x,y,z,l;

  bdx = width;
  bdy = height;

  if (a0 < 0) a0=0;
  else if (a0>=bdx) a0=bdx-1;

  if (a1<0)  a1=0;
  else  if (a1>=bdx)   a1=bdx-1;

  if (b0<0) b0=0;
  else if (b0>=bdy) b0=bdy-1;

  if (b1<0) 	b1=0;
  else if (b1>=bdy) b1=bdy-1;

  if (a0<a1) { sx = 1; dx = a1-a0; } else { sx = -1; dx = a0-a1; }
  if (b0<b1) { sy = 1; dy = b1-b0; } else { sy = -1; dy = b0-b1; }
  x=0; y=0;

  if (dx>=dy)
    {
      z = (-dx) / 2;
      while (abs(x) <= dx)
        {

          l =  (y+b0)*bdx+x+a0;

          igray[l] = value;

          x+=sx;
          z+=dy;
          if (z>0) { y+=sy; z-=dx; }

        }

    }
  else
    {
      z = (-dy) / 2;
      while (abs(y) <= dy) {

        l = (y+b0)*bdx+x+a0;
        igray[l] = value;

        y+=sy;
        z+=dx;
        if (z>0) { x+=sx; z-=dy; }
      }
    }

}


void draw_line(float *igray, int a0, int b0, int a1, int b1, float value, int width, int height)
{

  int bdx,bdy;
  int sx,sy,dx,dy,x,y,z,l;

  bdx = width;
  bdy = height;

  if (a0 < 0) a0=0;
  else if (a0>=bdx) a0=bdx-1;

  if (a1<0)  a1=0;
  else  if (a1>=bdx)   a1=bdx-1;

  if (b0<0) b0=0;
  else if (b0>=bdy) b0=bdy-1;

  if (b1<0) 	b1=0;
  else if (b1>=bdy) b1=bdy-1;

  if (a0<a1) { sx = 1; dx = a1-a0; } else { sx = -1; dx = a0-a1; }
  if (b0<b1) { sy = 1; dy = b1-b0; } else { sy = -1; dy = b0-b1; }
  x=0; y=0;

  if (dx>=dy)
    {
      z = (-dx) / 2;
      while (abs(x) <= dx)
        {

          l =  (y+b0)*bdx+x+a0;

          igray[l] = value;

          x+=sx;
          z+=dy;
          if (z>0) { y+=sy; z-=dx; }

        }

    }
  else
    {
      z = (-dy) / 2;
      while (abs(y) <= dy) {

        l = (y+b0)*bdx+x+a0;
        igray[l] = value;

        y+=sy;
        z+=dx;
        if (z>0) { x+=sx; z-=dy; }
      }
    }

}


void draw_square(float *igray, int a0, int b0, int w0, int h0, float value, int width, int height) //Mariano Rodríguez
{

        draw_line(igray,a0,b0,a0+w0,b0,value,width,height);
        draw_line(igray,a0,b0,a0,b0+h0,value,width,height);
        draw_line(igray,a0+w0,b0,a0+w0,b0+h0,value,width,height);
        draw_line(igray,a0,b0+h0,a0+w0,b0+h0,value,width,height);

}


void draw_parallelograms(float *igray, int* a, int* b, int* c, int* d, float value, int width, int height) //Mariano Rodríguez
{
        draw_line(igray,a[0],b[0],a[1],b[1],value,width,height);
        draw_line(igray,b[0],c[0],b[1],c[1],value,width,height);
        draw_line(igray,c[0],d[0],c[1],d[1],value,width,height);
        draw_line(igray,c[0],a[0],c[1],a[1],value,width,height);
}


#include "libNumerics/numerics.h"

void tiltedcoor2imagecoor_continous(float& x0, float& y0, float t, float Rtheta) //Mariano Rodríguez
{
    t = 1/t;
    float x1 = x0*t, y1 = y0;
    libNumerics::matrix<float> Rot(2,2);
    // Rot = [cos(Rtheta) -sin(Rtheta);sin(Rtheta) cos(Rtheta)];
    Rot(0,0) = cos(Rtheta); Rot(0,1) = -sin(Rtheta);
    Rot(1,0) = sin(Rtheta); Rot(1,1) = cos(Rtheta);



    libNumerics::matrix<float> vec(2,1);
    vec(0,0) = x1;
    vec(1,0) = y1;

    // Simulate rotation -> [x1;y1] = Rot*[x1;y1]
    vec = (Rot*vec);
    x1 = vec(0,0);
    y1 = vec(1,0);

    x0 = x1 ;
    y0 = y1 ;
}

//both angle and Rtheta in radians (not degrees)
//Mariano Rodríguez
void draw_circle_affine(float *igray,int w, int h, float x,float y, float angle, float radius,float t1, float t2, float Rtheta, float value)
{
    int discrete = 20;
    float theta = 0;

    float t = t1;

       // std::cout << t<< std::endl;

    angle = -angle;
    float mx = sin(angle)*radius , my = cos(angle)*radius;
    tiltedcoor2imagecoor_continous(mx,my,t,Rtheta);
    mx += x; my += y;
    draw_line(igray,x,y,mx,my,value,w,h);


    float _cx = 0, _cy = radius;

    tiltedcoor2imagecoor_continous(_cx,_cy,t,Rtheta);
    //compensate_affine_coor2(&_cy,&_cx,height,width,t,Rtheta);
    _cx += x;
    _cy += y;
float cx,cy;
    for (int i=1;i<=(discrete+1);i++)
    {

        // circle (r*sin(\theta),r*cos(\theta))
        cx = sin(theta)*radius;
        cy = cos(theta)*radius;

        // inverse of T_\t R_\Rtheta
        tiltedcoor2imagecoor_continous(cx,cy,t,Rtheta);
        //compensate_affine_coor2(&cy,&cx,height,width,t,Rtheta);

        // center around the point (x,y)
        cx += x;
        cy += y;

        draw_line(igray,_cx,_cy,cx,cy,value,w,h);

        _cx = cx;
        _cy = cy;

        theta += (2*M_PI/discrete);
    }

}

#define ABS(x)    (((x) > 0) ? (x) : (-(x)))
//both angle and Rtheta in radians (not degrees)
//Mariano Rodríguez
void draw_square_affine(float *igray,int w, int h, float x,float y, float angle, float radius,float t1, float t2, float Rtheta, float value)
{
    float t = t1;

       // std::cout << t<< std::endl;

    angle = -angle;

//    float side = (2*radius/sqrt(2))/2 ;
//    float mx = sin(angle)*side , my = cos(angle)*side;
//    tiltedcoor2imagecoor_continous(mx,my,t,Rtheta);
//    mx += x; my += y;
//    draw_line(igray,x,y,mx,my,value,w,h);

    float _cx = sin(M_PI/4+angle)*radius, _cy = cos(M_PI/4+angle)*radius;

    tiltedcoor2imagecoor_continous(_cx,_cy,t,Rtheta);
    //compensate_affine_coor2(&_cy,&_cx,height,width,t,Rtheta);
    _cx += x;
    _cy += y;
float cx,cy;
    for (int i=1;i<=4;i++)
    {

        float theta = i*M_PI/2 + M_PI/4 +angle;
        // circle (r*sin(\theta),r*cos(\theta))
        cx = sin(theta)*radius;
        cy = cos(theta)*radius;

        // inverse of T_\t R_\Rtheta
        tiltedcoor2imagecoor_continous(cx,cy,t,Rtheta);
        //compensate_affine_coor2(&cy,&cx,height,width,t,Rtheta);

        // center around the point (x,y)
        cx += x;
        cy += y;

        draw_line(igray,_cx,_cy,cx,cy,value,w,h);

        _cx = cx;
        _cy = cy;
    }
}


///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

/*


*/
/*
*/





/*

void _sign(float *u,float *v, int size)
{

	int i=0;

	for(i=0;i<size;i++){

		if (u[i]>0) v[i] = 1.0;
		else if (u[i]<0) v[i]=-1.0;
		else v[i]=0.0;
	}


}










void _multiple(float *u,float multiplier,int size)
{
   int i=0;
   float *ptru;

   ptru=&u[0];
   for(i=0;i<size;i++,ptru++)  *ptru=multiplier*(*ptru);

}


void _product(float *u,float *v,int size)
{
  int i;
  float *ptru,*ptrv;

  ptru=&u[0];
  ptrv=&v[0];
  for(i=0;i<size;i++,ptru++,ptrv++)   *ptru= *ptru*(*ptrv);
}



int _is_increasing(float *u,float tolerance, int size)
{

	int i=1;
	while (i < size && u[i] > (u[i-1] - tolerance))
	{
		i++;
	}

	if (i==size) return 1;
	else return 0;

}








void _offset(float *u,float offset,int size)
{
  int i=0;
  float *ptru;

  ptru=&u[0];
  for(i=0;i<size;i++,ptru++)  *ptru=*ptru + offset;

}














void _threshold(float *u, float *v,float valuem,float valueM, int size)
{

	int i;

	for(i=0;i<size;i++){

		if (u[i] >= valueM) 	v[i]= valueM;
		else if (u[i] <= valuem)  v[i]= valuem;
		else v[i] = u[i];

	}

}




void _absdif(float *u, float *v,int size)
{
	int i=0;

	for(i=0;i<size;i++)  u[i] = (float) fabsf( u[i] -  v[i]);
}









float *  _diag_gauss(int sflag,float std,int *size) //Create a 1d gauss kernel of standard deviation std  (megawave2)
{

	float *u,prec = 4.0,shift;
	double v;
	int n,i,flag;

	if (sflag) n=*size;
	else
		n = 1+2*(int)ceil((double)std*sqrt(prec*2.*log(10.)));

	u =(float *) malloc(n*sizeof(float));

	if (n==1)
		u[0]=1.0;
	else{

		shift = 0.5*(float)(n-1);

		for (i=(n+1)/2;i--;) {

			v = ((double)i - (double) shift)/(double)std;

			u[i] = u[n-1-i] = (float) exp(-2.0*0.5*v*v);  // 2.0 because distances are in the diagonal

		}
	}

	if (flag = _normalize(u,n)) {
		*size=n;
		return u;
	} else {
		printf("ERROR: mdSigGaussKernel: mdSigNormalize: normalization equals zero.\n Try to reduce std.\n");
	}
}









void  _quant(float *u,float *v,float lambda,int size)
{

	int i,n;
	float a=lambda/2;

	for(i=0;i<size;i++){

		n = (int) floorf(u[i] / a);
		if (n%2==0)
			v[i] = (float) n * a;
		else
			v[i] = (float) (n+1) * a;
	}

}



void  _projectquant(float *u,float *v,float lambda,int size)
{

	int i;
	float a=lambda/2;

	for(i=0;i<size;i++){

		if (v[i] < u[i] - a) v[i]=u[i] - a;
		else if (v[i] > u[i] + a) v[i]=u[i] + a;

	}

}

*/
