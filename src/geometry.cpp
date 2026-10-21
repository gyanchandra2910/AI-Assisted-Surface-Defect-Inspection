#include "geometry.hpp"
#include <stdexcept>
#include <vector>
namespace gfx {
Point Mat3::apply(Point p) const {
    return {a[0] * p.x + a[1] * p.y + a[2], a[3] * p.x + a[4] * p.y + a[5]};
}
Mat3 Mat3::operator*(const Mat3 &b) const {
    Mat3 c;
    c.a.fill(0);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                c.a[i * 3 + j] += a[i * 3 + k] * b.a[k * 3 + j];
    return c;
}
Mat3 Mat3::inverse() const {
    double d = a[0] * a[4] - a[1] * a[3];
    if (std::abs(d) < 1e-12)
        throw std::runtime_error("Singular transform");
    return {{a[4] / d, -a[1] / d, (a[1] * a[5] - a[4] * a[2]) / d, -a[3] / d, a[0] / d,
             (a[3] * a[2] - a[0] * a[5]) / d, 0, 0, 1}};
}
Mat3 imageTransform(double w, double h, Rect v, double zoom, double angle, Point pan) {
    double c = std::cos(angle), s = std::sin(angle);
    double rw = std::abs(c) * w + std::abs(s) * h, rh = std::abs(s) * w + std::abs(c) * h;
    double scale = std::min((v.x1 - v.x0) / rw, (v.y1 - v.y0) / rh) * zoom;
    Mat3 center{{1, 0, -w / 2, 0, 1, -h / 2, 0, 0, 1}},
        rotate{{scale * c, -scale * s, 0, scale * s, scale * c, 0, 0, 0, 1}};
    Mat3 translate{{1, 0, (v.x0 + v.x1) / 2 + pan.x, 0, 1, (v.y0 + v.y1) / 2 + pan.y, 0, 0, 1}};
    return translate * rotate * center;
}
bool liangBarsky(Rect r, Point &a, Point &b) {
    double dx = b.x - a.x, dy = b.y - a.y, t0 = 0, t1 = 1;
    double p[4] = {-dx, dx, -dy, dy}, q[4] = {a.x - r.x0, r.x1 - a.x, a.y - r.y0, r.y1 - a.y};
    for (int i = 0; i < 4; i++) {
        if (std::abs(p[i]) < 1e-12) {
            if (q[i] < 0)
                return false;
        } else {
            double t = q[i] / p[i];
            if (p[i] < 0)
                t0 = std::max(t0, t);
            else
                t1 = std::min(t1, t);
            if (t0 > t1)
                return false;
        }
    }
    Point start = a;
    a = {start.x + t0 * dx, start.y + t0 * dy};
    b = {start.x + t1 * dx, start.y + t1 * dy};
    return true;
}
bool pointInPolygon(Point p, const std::vector<Point> &poly) {
    bool inside = false;
    if (poly.empty())
        return false;
    for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
        auto a = poly[i], b = poly[j];
        if (((a.y > p.y) != (b.y > p.y)) && (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x))
            inside = !inside;
    }
    return inside;
}
} // namespace gfx
