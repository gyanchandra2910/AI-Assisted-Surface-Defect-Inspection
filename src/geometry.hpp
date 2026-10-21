#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
namespace gfx {
struct Point {
    double x = 0, y = 0;
};
struct Rect {
    double x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    bool contains(Point p) const {
        return p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1;
    }
};
// Row-major homogeneous affine matrix; pixels have their origin at the top left.
struct Mat3 {
    std::array<double, 9> a{1, 0, 0, 0, 1, 0, 0, 0, 1};
    Point apply(Point p) const;
    Mat3 inverse() const;
    Mat3 operator*(const Mat3 &b) const;
};
Mat3 imageTransform(double width, double height, Rect viewport, double zoom, double angle,
                    Point pan);
bool liangBarsky(Rect r, Point &a, Point &b);
bool pointInPolygon(Point p, const std::vector<Point> &polygon);
// All results are tagged. Only the currently requested source/frame may be displayed.
inline bool matchesSource(uint64_t requested, uint64_t completed) {
    return requested == completed;
}
} // namespace gfx
