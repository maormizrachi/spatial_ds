#ifndef SPHERE_HPP
#define SPHERE_HPP

#ifdef USE_VCL_VECTORIZATION
    #include <vectorclass.h>
#endif // USE_VCL_VECTORIZATION

    #include <mpi_utils/serialize/Serializer.hpp>

#include <iostream>

#define DIM 3

#define TOLERANCE 1e-12

template<typename T>
class Sphere
                : public Serializable
{
public:
    T center;
    typename T::coord_type radius;

    template<typename U> 
    Sphere(const U &center, typename T::coord_type radius): radius(radius)
    {
        this->center[0] = center[0];
        this->center[1] = center[1];
        this->center[2] = center[2];
    }

    Sphere(): Sphere(T(), typename T::coord_type()){};

    template<typename U>
    bool contains(const U &point) const;

    friend inline std::ostream &operator<<(std::ostream &stream, const Sphere<T> &sphere)
    {
        return stream << "Sphere(" << sphere.center << ", " << sphere.radius << ")";
    }

        force_inline size_t load(const Serializer *serializer, std::size_t byteOffset) override
        {
            size_t bytes = 0;
            bytes += serializer->extract(this->center, byteOffset);
            bytes += serializer->extract(this->radius, byteOffset + bytes);
            return bytes;
        }

        force_inline size_t dump(Serializer *serializer) const override
        {
            size_t bytes = 0;
            bytes += serializer->insert(this->center);
            bytes += serializer->insert(this->radius);
            return bytes;
        }
};

template<typename T>
template<typename U>
bool Sphere<T>::contains(const U &point) const
{
    typename T::coord_type distance2 = 0, radius2 = 0;
    #ifdef USE_VCL_VECTORIZATION
        Vec4d diff(point[0] - this->center[0], point[1] - this->center[1], point[2] - this->center[2], this->radius);
        Vec4d distanceSquared = diff * diff;
        distance2 = distanceSquared[0] + distanceSquared[1] + distanceSquared[2];
        radius2 = distanceSquared[3];
    #else // USE_VCL_VECTORIZATION
        for(int i = 0; i < DIM; i++)
        {
            double _distance = (point[i] - this->center[i]);
            distance2 += _distance * _distance;
        }
        radius2 = (this->radius * this->radius);
    #endif // USE_VCL_VECTORIZATION
    return distance2 <= (1 + TOLERANCE) * radius2;
}

#endif // SPHERE_HPP