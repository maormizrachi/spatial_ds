#ifndef DISTRIBUTED_OCTTREE_HPP
#define DISTRIBUTED_OCTTREE_HPP

#include <cassert>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <algorithm>

#include <vector>
#include <utility>
#include <array>
#include <bitset>
#include <boost/container/flat_set.hpp>
#include <mpi.h>
#include <spatial_ds/OctTree/OctTree.hpp>
#include "RankedValue.hpp"


#ifdef DEBUG_MODE
#include <iostream>
#endif // DEBUG_MODE

#define MAX_DIRECTIONS_SIZE (MAX_DEPTH + 1)

#define MAX_RANKS_FOR_LEAF_DEFAULT 4

template<typename T, int max_ranks_per_leaf, int max_directions_size>
using DistributedOctNode = typename OctTree<OctTreeRankedValue<T, max_ranks_per_leaf, max_directions_size>>::OctTreeNode;

template<typename T, int max_ranks_per_leaf = MAX_RANKS_FOR_LEAF_DEFAULT>
class DistributedOctTree
{
public:
    using RankedValue = OctTreeRankedValue<T, max_ranks_per_leaf, MAX_DIRECTIONS_SIZE>;
    using DistributedOctTreeNode = DistributedOctNode<T, max_ranks_per_leaf, MAX_DIRECTIONS_SIZE>;

public:
    DistributedOctTree(const OctTree<T> *tree, bool detailedNodeInfo = false, const MPI_Comm &comm = MPI_COMM_WORLD);

    ~DistributedOctTree(){delete this->octTree;};

    #ifdef DEBUG_MODE
    void print() const{this->octTree->print();};
    #endif // DEBUG_MODE

    boost::container::flat_set<int> getIntersectingRanks(const Sphere<T> &sphere) const;
    
    inline boost::container::flat_set<int> getIntersectingRanks(const T &center, const typename T::coord_type radius) const{return this->getIntersectingRanks(Sphere<T>(center, radius));};
    
    int getDepth() const{return this->octTree->getDepth();};
    
    OctTree<RankedValue> *getOctTree(){return this->octTree;};

    const OctTree<RankedValue> *getOctTree() const{return this->octTree;};

    std::vector<T> getRankValues(int _rank) const;

    inline std::vector<T> getMyValues() const{return this->getRankValues(this->rank);};

    std::vector<BoundingBox<T>> getRankBoundingBoxes(int _rank) const;

    std::vector<BoundingBox<T>> getRankBoundingBoxesUnique(int _rank) const;

    inline std::vector<BoundingBox<T>> getMyBoundingBoxes() const{return this->getRankBoundingBoxes(this->rank);};

    inline std::vector<BoundingBox<T>> getMyBoundingBoxesUnique() const{return this->getRankBoundingBoxesUnique(this->rank);};

    std::vector<std::vector<BoundingBox<T>>> getBoundingBoxesOfRanks(void) const;

    std::vector<BoundingBox<T>> getBigBoundingBoxesOfRanks(void) const;

    std::vector<std::vector<direction_t>> getRankDirections(int _rank) const;

    template<typename U>
    std::vector<rank_t> GetRanksOfPoint(const U &point) const;

    inline std::vector<const DistributedOctTreeNode*> getRankNodes(int _rank) const
    {
        return this->getValuesIf([](const DistributedOctTreeNode *node){return node->value.owners.empty();}, 
                            [&_rank](const DistributedOctTreeNode *node){return std::find(node->value.owners.cbegin(), node->value.owners.cend(), _rank) != node->value.owners.cend();});
    }

    inline std::vector<std::vector<direction_t>> getMyDirections() const{return this->getRankDirections(this->rank);};

    inline std::vector<const DistributedOctTreeNode*> getMyNodes() const{return this->getRankNodes(this->rank);};

    template<typename U>
    std::vector<std::pair<typename T::coord_type, typename T::coord_type>> getClosestFurthestPointsByRanks(const U &point) const;

    #ifdef DEBUG_MODE
    inline bool validate() const{if(this->octTree != nullptr) return this->validateHelper(this->octTree->getRoot()); return true;};
    #endif // DEBUG_MODE

    std::vector<const DistributedOctTreeNode*> getValuesIf(const std::function<bool(const DistributedOctTreeNode*)> &ifOpenFunction, const std::function<bool(const DistributedOctTreeNode*)> &ifAddValueFunction) const;

private:
    OctTree<RankedValue> *octTree = nullptr;
    MPI_Comm comm;
    int rank, size;
    bool detailedNodeInfo; // whether or not to save detailed info on the leaf nodes
    size_t treeSize;

    void buildTreeHelper(DistributedOctTreeNode *newNode, const typename OctTree<T>::OctTreeNode *node, std::vector<direction_t> &directionsInMyTree);

    // Level-synchronous build without detailed node info: the same tree as buildTreeHelper, with one MPI_Allgather
    // per tree level instead of one per internal node.
    void buildTreeBatched(const OctTree<T> *tree);

    // Structure and owners of two trees agree node by node.
    static bool sameTree(const DistributedOctTreeNode *a, const DistributedOctTreeNode *b);

    void buildTree(const OctTree<T> *tree);

    #ifdef DEBUG_MODE
    bool validateHelper(const DistributedOctTreeNode *node) const;
    #endif // DEBUG_MODE
};

template<typename T, int max_ranks_per_leaf>
DistributedOctTree<T, max_ranks_per_leaf>::DistributedOctTree(const OctTree<T> *tree, bool detailedNodeInfo, const MPI_Comm &comm): comm(comm), detailedNodeInfo(detailedNodeInfo)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);
    this->buildTree(tree);
}

template<typename T, int max_ranks_per_leaf>
void DistributedOctTree<T, max_ranks_per_leaf>::buildTreeHelper(DistributedOctTreeNode *newNode, const typename OctTree<T>::OctTreeNode *node, std::vector<direction_t> &directionsInMyTree)
{
    assert(newNode != nullptr);
    unsigned char valueToSend = 0; // assumes `CHILDREN` is 8. this variable contains 1 in the `i`th bit iff child `i` exists
    if(node != nullptr)
    {
        for(int i = 0; i < CHILDREN; i++)
        {
            bool bit = (node->children[i] != nullptr || (node->isLeaf and newNode->getChildNumberContaining(node->value) == i));
            valueToSend |= (bit << i);
        }
    }

    std::vector<unsigned char> childBuff(this->size);
    MPI_Allgather(&valueToSend, 1, MPI_UNSIGNED_CHAR, &childBuff[0], 1, MPI_BYTE, this->comm);

    for(int i = 0; i < CHILDREN; i++)
    {
        bool recursiveBuild = false;
        int containingValue = UNDEFINED_OWNER; // who has the `i`th child
        std::vector<int> ranks_containing_child;

        for(int _rank = 0; _rank < this->size; _rank++)
        {
            if(((childBuff[_rank] >> i) & 0x1) == 1)
            {
                ranks_containing_child.push_back(_rank);

                if(containingValue == UNDEFINED_OWNER)
                {
                    // first rank to announce it holds the value
                    containingValue = _rank;
                }
                else
                {
                    // more than one child has a point in this route of the tree, so we continue to split recursively
                    recursiveBuild = true;
                }
            }
        }
        // if somebody holds the value
        if(containingValue != UNDEFINED_OWNER)
        {
            // someone holds the `i`th child
            this->treeSize++;
            newNode->createChild(i); // creates the child in my own tree
            if(recursiveBuild and (ranks_containing_child.size() > max_ranks_per_leaf))
            {
                // there are several holders, call recursive build (until we reach one holder)
                // determine what's the next node in my own tree to continue the recursive build with
                // this node might be null, if I don't have any nodes this depth in the tree
                const typename OctTree<T>::OctTreeNode *nextNode = nullptr;

                bool advancedNode = false; // if went down to a child of current node
                if(node != nullptr)
                {
                    if(node->isLeaf)
                    {
                        nextNode = newNode->children[i]->boundingBox.contains(node->value)? node : nullptr;
                    }
                    else
                    {
                        advancedNode = true;
                        directionsInMyTree.push_back(i);
                        nextNode = node->children[i];
                    }
                }
                else
                {
                    nextNode = nullptr;
                }
                // continue recursively
                this->buildTreeHelper(newNode->children[i], nextNode, directionsInMyTree);
                if(advancedNode)
                {
                    directionsInMyTree.pop_back();
                }
                // several owners
                newNode->children[i]->value.owners.resize(0);
                newNode->children[i]->isLeaf = false;
            }
            else
            {
                if((ranks_containing_child.size() == 1) or (this->detailedNodeInfo))
                {
                    // there is only one holder, set its owner field
                    if(this->detailedNodeInfo)
                    {
                        if(this->rank == containingValue)
                        {
                            const typename OctTree<T>::OctTreeNode *childNode;
                            if(node->isLeaf)
                            {
                                childNode = node;
                            }
                            else
                            {
                                childNode = node->children[i];
                                directionsInMyTree.push_back(i);
                            }
            
                            // copy value, and directions
                            std::memcpy(&newNode->children[i]->value.value, &childNode->value, sizeof(T));
                            std::memcpy(newNode->children[i]->value.directions, directionsInMyTree.data(), sizeof(direction_t) * directionsInMyTree.size());
                            newNode->children[i]->value.directions[directionsInMyTree.size()] = PATH_END_DIRECTION;

                            if(!node->isLeaf)
                            {
                                directionsInMyTree.pop_back();
                            }
                        }
                        MPI_Bcast(&newNode->children[i]->value.value, sizeof(T), MPI_BYTE, containingValue, this->comm);
                        MPI_Bcast(newNode->children[i]->value.directions, sizeof(direction_t) * MAX_DIRECTIONS_SIZE, MPI_BYTE, containingValue, this->comm);
                    }

                    newNode->children[i]->value.owners = {containingValue};
                }
                else
                {
                    // several owners
                    newNode->children[i]->value.owners.resize(ranks_containing_child.size());
                    std::copy(ranks_containing_child.cbegin(), ranks_containing_child.cend(), newNode->children[i]->value.owners.begin());
                }
                newNode->children[i]->isLeaf = true;
            }
        }
        else
        {
            this->treeSize++;
            newNode->children[i] = nullptr;
        }
    }
}

#ifdef DEBUG_MODE
template<typename T, int max_ranks_per_leaf>
bool DistributedOctTree<T, max_ranks_per_leaf>::validateHelper(const DistributedOctTreeNode *node) const
{
    if(node == nullptr)
    {
        return true;
    }
    bool hasChildren = false;
    for(int i = 0; i < CHILDREN; i++)
    {
        if(node->children[i] != nullptr)
        {
            hasChildren = true;
            break;
        }
    }
    if(hasChildren)
    {
        assert(node->isLeaf);
    }
    for(int i = 0; i < CHILDREN; i++)
    {
        assert(this->validateHelper(node->children[i]));
    }
    return true;
}
#endif // DEBUG_MODE

template<typename T, int max_ranks_per_leaf>
void DistributedOctTree<T, max_ranks_per_leaf>::buildTree(const OctTree<T> *tree)
{
    assert(this->octTree == nullptr);
    if(tree == nullptr or tree->getRoot() == nullptr)
    {
        return;
    }
    std::vector<direction_t> directions;
    directions.reserve(MAX_DIRECTIONS_SIZE);
    // tree->print();

    RankedValue ll, ur;

    // RICH_DISTRIBUTED_OCTTREE_BATCHED: 1 (default) builds level by level, 0 recursively (one collective per internal
    // node), 2 builds both and fails unless they agree.  Detailed node info always builds recursively (it broadcasts
    // per leaf).  Agreed on this->comm at every construction, which every rank of the communicator makes together.
    int mode = 1;
    if(!this->detailedNodeInfo)
    {
        char const* const value = std::getenv("RICH_DISTRIBUTED_OCTTREE_BATCHED");
        if(value != nullptr && value[0] != '\0')
            mode = (value[0] >= '0' && value[0] <= '2' && value[1] == '\0') ? value[0] - '0' : -1;
        int extrema[2] = {mode, -mode};
        MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_INT, MPI_MAX, this->comm);
        if(extrema[0] != -extrema[1] || extrema[0] < 0)
            throw std::invalid_argument("RICH_DISTRIBUTED_OCTTREE_BATCHED must be 0, 1 or 2 on every rank");
        mode = extrema[0];
    }
    else
        mode = 0;

    this->octTree = new OctTree<RankedValue>(tree->getLL(), tree->getUR());
    this->treeSize = 0;
    if(mode == 0)
    {
        this->buildTreeHelper(this->octTree->getRoot(), tree->getRoot(), directions);
        return;
    }
    this->buildTreeBatched(tree);
    if(mode == 2)
    {
        OctTree<RankedValue> *const batched = this->octTree;
        size_t const batchedSize = this->treeSize;
        this->octTree = new OctTree<RankedValue>(tree->getLL(), tree->getUR());
        this->treeSize = 0;
        this->buildTreeHelper(this->octTree->getRoot(), tree->getRoot(), directions);
        bool const same = batchedSize == this->treeSize && sameTree(batched->getRoot(), this->octTree->getRoot());
        delete this->octTree;
        this->octTree = batched;
        if(!same)
            throw std::logic_error("Batched distributed octree differs from the recursive build");
    }
}

template<typename T, int max_ranks_per_leaf>
bool DistributedOctTree<T, max_ranks_per_leaf>::sameTree(const DistributedOctTreeNode *a, const DistributedOctTreeNode *b)
{
    if(a == nullptr || b == nullptr)
        return a == b;
    if(a->isLeaf != b->isLeaf || a->value.owners.size() != b->value.owners.size() ||
       !std::equal(a->value.owners.cbegin(), a->value.owners.cend(), b->value.owners.cbegin()))
        return false;
    for(int i = 0; i < CHILDREN; i++)
        if(!sameTree(a->children[i], b->children[i]))
            return false;
    return true;
}

template<typename T, int max_ranks_per_leaf>
void DistributedOctTree<T, max_ranks_per_leaf>::buildTreeBatched(const OctTree<T> *tree)
{
    // One entry per node still to split, in the same order on every rank (the order the decisions produce): the
    // node in the distributed tree and this rank's matching local node, or nullptr.  Each node is decided exactly
    // as in buildTreeHelper, from the child bits every rank reports for it.
    using LocalNode = typename OctTree<T>::OctTreeNode;
    std::vector<std::pair<DistributedOctTreeNode*, const LocalNode*>> open{{this->octTree->getRoot(), tree->getRoot()}};
    std::vector<std::pair<DistributedOctTreeNode*, const LocalNode*>> next;
    std::vector<unsigned char> mine, gathered;
    std::vector<int> ranks_containing_child;
    while(!open.empty())
    {
        size_t const count = open.size();
        mine.assign(count, 0);
        for(size_t k = 0; k < count; k++)
        {
            DistributedOctTreeNode *const newNode = open[k].first;
            const LocalNode *const node = open[k].second;
            if(node == nullptr)
                continue;
            unsigned char valueToSend = 0;
            for(int i = 0; i < CHILDREN; i++)
            {
                bool bit = (node->children[i] != nullptr || (node->isLeaf and newNode->getChildNumberContaining(node->value) == i));
                valueToSend |= (bit << i);
            }
            mine[k] = valueToSend;
        }
        if(count > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("Distributed octree level too wide for one MPI_Allgather");
        gathered.resize(count * static_cast<size_t>(this->size));
        MPI_Allgather(mine.data(), static_cast<int>(count), MPI_UNSIGNED_CHAR, gathered.data(),
                      static_cast<int>(count), MPI_UNSIGNED_CHAR, this->comm);
        next.clear();
        for(size_t k = 0; k < count; k++)
        {
            DistributedOctTreeNode *const newNode = open[k].first;
            const LocalNode *const node = open[k].second;
            for(int i = 0; i < CHILDREN; i++)
            {
                ranks_containing_child.clear();
                for(int _rank = 0; _rank < this->size; _rank++)
                    if(((gathered[static_cast<size_t>(_rank) * count + k] >> i) & 0x1) == 1)
                        ranks_containing_child.push_back(_rank);
                this->treeSize++;
                if(ranks_containing_child.empty())
                {
                    newNode->children[i] = nullptr;
                    continue;
                }
                newNode->createChild(i);
                if(ranks_containing_child.size() > static_cast<size_t>(max_ranks_per_leaf))
                {
                    // Several holders: split further (the local node may be null, or a leaf outside this child).
                    const LocalNode *nextNode = nullptr;
                    if(node != nullptr)
                        nextNode = node->isLeaf ? (newNode->children[i]->boundingBox.contains(node->value) ? node : nullptr)
                            : node->children[i];
                    next.emplace_back(newNode->children[i], nextNode);
                    newNode->children[i]->value.owners.resize(0);
                    newNode->children[i]->isLeaf = false;
                }
                else
                {
                    newNode->children[i]->value.owners.resize(ranks_containing_child.size());
                    std::copy(ranks_containing_child.cbegin(), ranks_containing_child.cend(),
                              newNode->children[i]->value.owners.begin());
                    newNode->children[i]->isLeaf = true;
                }
            }
        }
        open.swap(next);
    }
}

template<typename T, int max_ranks_per_leaf>
boost::container::flat_set<int> DistributedOctTree<T, max_ranks_per_leaf>::getIntersectingRanks(const Sphere<T> &sphere) const
{
    boost::container::flat_set<int> ranks;
    for(const RankedValue &point : this->octTree->range(Sphere<RankedValue>(sphere.center, sphere.radius)))
    {
        size_t numOwners = point.owners.size();
        for(size_t i = 0; i < numOwners; i++)
        {
            ranks.insert(point.owners[i]);
        }
    }
    return ranks;
}

template<typename T, int max_ranks_per_leaf>
template<typename U>
std::vector<rank_t> DistributedOctTree<T, max_ranks_per_leaf>::GetRanksOfPoint(const U &point) const
{
    const RankedValue &value = this->octTree->GetContainingNodeValue(point);
    std::vector<rank_t> ranks;
    for(const rank_t &rank : value.owners)
    {
        ranks.push_back(rank);
    }
    return ranks;
}

template<typename T, int max_ranks_per_leaf>
template<typename U>
std::vector<std::pair<typename T::coord_type, typename T::coord_type>> DistributedOctTree<T, max_ranks_per_leaf>::getClosestFurthestPointsByRanks(const U &point) const
{
    const typename T::coord_type &maxVal = std::numeric_limits<typename T::coord_type>::max();
    const typename T::coord_type &minVal = std::numeric_limits<typename T::coord_type>::lowest();
    
    std::pair<T, T> initialPair = std::make_pair<T, T>(T(maxVal, maxVal, maxVal), T(minVal, minVal, minVal));
    std::vector<std::pair<typename T::coord_type, typename T::coord_type>> distances(this->size, {maxVal, minVal});

    std::vector<const DistributedOctTreeNode*> nodes;
    nodes.reserve(this->getDepth() * CHILDREN);
    if(this->octTree->getRoot() != nullptr)
    {
        nodes.push_back(this->octTree->getRoot());
    }

    T closestPoint, furthestPoint;
    while(!nodes.empty())
    {
        const DistributedOctTreeNode *node = nodes.back();
        nodes.pop_back();

        if(!node->isLeaf)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                if(node->children[i] != nullptr)
                {
                    nodes.push_back(node->children[i]);
                }
            }
            continue;
        }
        // node is a value node
        closestPoint = node->boundingBox.closestPoint(point);
        furthestPoint = node->boundingBox.furthestPoint(point);
        typename T::coord_type closestDist = 0, furthestDist = 0;
        #ifdef USE_VCL_VECTORIZATION
            Vec8d diff(closestPoint[0] - point[0], closestPoint[1] - point[1], closestPoint[2] - point[2],
                        furthestPoint[0] - point[0], furthestPoint[1] - point[1], furthestPoint[2] - point[2],
                        0, 0);
            Vec8d dist = diff * diff;
            closestDist = dist[0] + dist[1] + dist[2];
            furthestDist = dist[3] + dist[4] + dist[5];
        #else // USE_VCL_VECTORIZATION
            closestDist += (closestPoint[0] - point[0]) * (closestPoint[0] - point[0]);
            furthestDist += (furthestPoint[0] - point[0]) * (furthestPoint[0] - point[0]);
            closestDist += (closestPoint[1] - point[1]) * (closestPoint[1] - point[1]);
            furthestDist += (furthestPoint[1] - point[1]) * (furthestPoint[1] - point[1]);
            closestDist += (closestPoint[2] - point[2]) * (closestPoint[2] - point[2]);
            furthestDist += (furthestPoint[2] - point[2]) * (furthestPoint[2] - point[2]);
        #endif // USE_VCL_VECTORIZATION
        size_t numOwners = node->value.owners.size();

        for(size_t i = 0; i < numOwners; i++)
        {
            int owner = node->value.owners[i];
            if(distances[owner].first > closestDist)
            {
                distances[owner].first = closestDist;
            }
            if(distances[owner].second < furthestDist)
            {
                distances[owner].second = furthestDist;
            }
        }
    }
    return distances;
}


template<typename T, int max_ranks_per_leaf>
std::vector<T> DistributedOctTree<T, max_ranks_per_leaf>::getRankValues(int _rank) const
{
    std::vector<T> values;
    std::vector<const DistributedOctTreeNode*> nodes;
    nodes.reserve(this->getDepth() * CHILDREN);

    nodes.push_back(this->octTree->getRoot());

    while(!nodes.empty())
    {
        const DistributedOctTreeNode *node = nodes.back();
        nodes.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        size_t numOwners = node->value.owners.size();

        if(numOwners == 0)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                nodes.push_back(node->children[i]);
            }
        }
        else
        {
            if(std::find(node->value.owners.cbegin(), node->value.owners.cend(), _rank) != node->value.owners.cend())
            {
                // is an owner
                values.emplace_back(node->value.value);
            }
        }
    }
    return values;
}

template<typename T, int max_ranks_per_leaf>
std::vector<BoundingBox<T>> DistributedOctTree<T, max_ranks_per_leaf>::getRankBoundingBoxes(int _rank) const
{
    std::vector<BoundingBox<T>> boxes;
    std::vector<const DistributedOctTreeNode*> nodes;
    nodes.reserve(this->getDepth() * CHILDREN);

    nodes.push_back(this->octTree->getRoot());

    while(!nodes.empty())
    {
        const DistributedOctTreeNode *node = nodes.back();
        nodes.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        size_t numOwners = node->value.owners.size();

        if(numOwners == 0)
        {
            // several owners
            for(int i = 0; i < CHILDREN; i++)
            {
                nodes.push_back(node->children[i]);
            }
        }
        else
        {
            if(std::find(node->value.owners.cbegin(), node->value.owners.cend(), _rank) != node->value.owners.cend())
            {
                // is an owner
                boxes.emplace_back(BoundingBox<T>(node->boundingBox.getLL(), node->boundingBox.getUR()));
            }
        }
    }
    return boxes;
}

template<typename T, int max_ranks_per_leaf>
std::vector<BoundingBox<T>> DistributedOctTree<T, max_ranks_per_leaf>::getRankBoundingBoxesUnique(int _rank) const
{
    std::vector<BoundingBox<T>> boxes;
    std::vector<const DistributedOctTreeNode*> nodes;
    nodes.reserve(this->getDepth() * CHILDREN);

    nodes.push_back(this->octTree->getRoot());

    while(!nodes.empty())
    {
        const DistributedOctTreeNode *node = nodes.back();
        nodes.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        size_t numOwners = node->value.owners.size();

        if(numOwners == 0)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                nodes.push_back(node->children[i]);
            }
        }
        else
        {
            if(*std::min_element(node->value.owners.cbegin(), node->value.owners.cend()) == _rank)
            {
                // is an owner
                boxes.emplace_back(BoundingBox<T>(node->boundingBox.getLL().value, node->boundingBox.getUR().value));
            }
        }
    }
    return boxes;
}

template<typename T, int max_ranks_per_leaf>
std::vector<std::vector<BoundingBox<T>>> DistributedOctTree<T, max_ranks_per_leaf>::getBoundingBoxesOfRanks(void) const
{
    std::vector<std::vector<BoundingBox<T>>> boxes(this->size);
    std::vector<const DistributedOctTreeNode*> nodes;
    nodes.reserve(this->getDepth() * CHILDREN);

    nodes.push_back(this->octTree->getRoot());

    while(!nodes.empty())
    {
        const DistributedOctTreeNode *node = nodes.back();
        nodes.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        size_t numOwners = node->value.owners.size();

        if(numOwners == 0)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                nodes.push_back(node->children[i]);
            }
        }
        else
        {
            for(int _rank : node->value.owners)
            {
                boxes[_rank].push_back(node->boundingBox);
            }
        }
    }
    return boxes;
}

template<typename T, int max_ranks_per_leaf>
std::vector<BoundingBox<T>> DistributedOctTree<T, max_ranks_per_leaf>::getBigBoundingBoxesOfRanks(void) const
{
    const BoundingBox<T> &rootBB = this->octTree->getRoot()->boundingBox;
    std::vector<std::pair<T, T>> boxes(this->size, {rootBB.getUR(), rootBB.getLL()});
    std::vector<const DistributedOctTreeNode*> nodes;
    nodes.reserve(this->getDepth() * CHILDREN);

    nodes.push_back(this->octTree->getRoot());

    while(!nodes.empty())
    {
        const DistributedOctTreeNode *node = nodes.back();
        nodes.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        size_t numOwners = node->value.owners.size();

        if(numOwners == 0)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                nodes.push_back(node->children[i]);
            }
        }
        else
        {
            const T &BB_ll = node->boundingBox.getLL();
            const T &BB_ur = node->boundingBox.getUR();
            for(int _rank : node->value.owners)
            {
                boxes[_rank].first[0] = std::min(boxes[_rank].first[0], BB_ll[0]);
                boxes[_rank].second[0] = std::max(boxes[_rank].second[0], BB_ur[0]);
                boxes[_rank].first[1] = std::min(boxes[_rank].first[1], BB_ll[1]);
                boxes[_rank].second[1] = std::max(boxes[_rank].second[1], BB_ur[1]);
                boxes[_rank].first[2] = std::min(boxes[_rank].first[2], BB_ll[2]);
                boxes[_rank].second[2] = std::max(boxes[_rank].second[2], BB_ur[2]);
            }
        }
    }
    std::vector<BoundingBox<T>> returnBoundingBoxes;
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        returnBoundingBoxes.push_back(BoundingBox<T>(boxes[_rank].first, boxes[_rank].second));
    }
    return returnBoundingBoxes;
}


template<typename T, int max_ranks_per_leaf>
std::vector<std::vector<direction_t>> DistributedOctTree<T, max_ranks_per_leaf>::getRankDirections(int _rank) const
{
    std::vector<std::vector<direction_t>> directions;
    std::vector<const DistributedOctTreeNode*> nodes;
    nodes.reserve(this->getDepth() * CHILDREN);

    nodes.push_back(this->octTree->getRoot());

    while(!nodes.empty())
    {
        const DistributedOctTreeNode *node = nodes.back();
        nodes.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        size_t numOwners = node->value.owners.size();

        if(numOwners == 0)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                nodes.push_back(node->children[i]);
            }
        }
        else
        {
            if(std::find(node->value.owners.cbegin(), node->value.owners.cend(), _rank) != node->value.owners.cend())
            {
                // is an owner
                size_t directionsLength = std::distance(node->value.directions, std::find(node->value.directions, node->value.directions + MAX_DIRECTIONS_SIZE, PATH_END_DIRECTION)) + 1; // + 1 for the `PATH_END_DIRECTION`
                directions.push_back(std::vector(node->value.directions, node->value.directions + directionsLength));
            }
        }
    }
    return directions;
}

template<typename T, int max_ranks_per_leaf>
std::vector<const typename DistributedOctTree<T, max_ranks_per_leaf>::DistributedOctTreeNode*>
    DistributedOctTree<T, max_ranks_per_leaf>::getValuesIf(const std::function<bool(const DistributedOctTreeNode*)> &ifOpenFunction, const std::function<bool(const DistributedOctTreeNode*)> &ifAddValueFunction) const
{
    std::vector<const DistributedOctTreeNode*> nodes = {this->octTree->getRoot()};
    nodes.reserve(this->getDepth() * CHILDREN);

    std::vector<const DistributedOctTreeNode*> result;

    while(not nodes.empty())
    {
        const DistributedOctTreeNode *node = nodes.back();
        nodes.pop_back();
        if(node == nullptr)
        {
            continue;
        }
        if(node->isLeaf)
        {
            if(ifAddValueFunction(node))
            {
                result.push_back(node);
            }
            continue;
        }
        if(ifOpenFunction(node))
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                nodes.push_back(node->children[i]);
            }
        }
    }
    return result;
}


#endif // DISTRIBUTED_OCTTREE_HPP