#ifndef _LIBRARY_H_
#define _LIBRARY_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
// #include <unistd.h>
#include <float.h>


void draw_line(double *igray, int a0, int b0, int a1, int b1, float value, int width, int height);
void draw_line(float *igray, int a0, int b0, int a1, int b1, float value, int width, int height);
// void draw_circle(float *igray, int pi,int pj,float radius, float value, int width, int height);

/**
 * @brief Draws a rectangle starting from the point \f$(a0,b0) \in [1,width]\times[1,height] \f$. The three other corners of the rectangle are: \f$(a0+w0,b0), (a0,b0+h0), (a0+w0,b0+h0).\f$
 * @param igray A one channel image onto which a rectangle is to be created.
 * @param (a0,b0) Image coordinates of the starting point of the rectangle
 * @param w0 width of the rectangle
 * @param h0 height of the rectangle
 * @param value Intensity of the lines belonging to the rectangle
 * @param width Image width
 * @param height Image height
 * @author Mariano Rodríguez
 */
void draw_square(float *igray, int a0, int b0, int w0, int h0, float value, int width, int height);

/**
 * @brief Draws circle following an affine transformation (which are in general ellipses).
 * @param igray A one channel image onto which an ellipse is to be created.
 * @param w Image width
 * @param h Image height
 * @param (x,y) center of the ellipse
 * @param angle Angle from which the keypoint descriptor starts. A line from the center to the ellipse is draw following that angle.
 * @param radius The radius of the circle before being transformed into an ellipse
 * @param t1 Tilt in the x direction that was applied after a rotation \f$ R_{Rtheta} \f$
 * @param t2 Tilt in the y direction that was applied after a rotation \f$ R_{Rtheta} \f$
 * @param Rtheta Angle in radians of the rotation.
 * @param value Intensity of the ellipse borders
 * @author Mariano Rodríguez
 */
void draw_circle_affine(float *igray, int w, int h, float x, float y, float angle, float radius, float t1, float t2, float Rtheta, float value);


void draw_square_affine(float *igray,int w, int h, float x,float y, float angle, float radius,float t1, float t2, float Rtheta, float value);
#endif // _LIBRARY_H_
