export default class ShapeUtils {

    /**
     * Broadcast two shapes, return broadcasted shape
     */
    static broadcastShapes(a: number[], b: number[]): number[] {
        let rank = Math.max(a.length, b.length);
        let result = new Array(rank); //uninitialized

        let aIndex = a.length - 1;
        let bIndex = b.length - 1;

        for (let dim = rank - 1; dim >= 0; dim--) {
            let left = aIndex >= 0 ? a[aIndex] : 1;
            let right = bIndex >= 0 ? b[bIndex] : 1;

            if (left !== 1 && right !== 1 && left !== right) {
                throw new Error('cannot broadcast shapes. not compatible');
            }

            result[dim] = Math.max(left, right);
            aIndex--;
            bIndex--;
        }

        return result;
    }

    /**
     * Get the shape that must be reshaped to match result's rank
     */
    static getBroadcastedShape(input: number[], result: number[]): number[] {
        if (input.length >= result.length) {
            return input;
        }

        let newShape = input.slice();
        for (let i = 0; i < result.length - input.length; i++) {
            newShape.unshift(1);
        }
        return newShape;
    }

    static getLength(shape: number[]): number {
        let mul = 1;
        for (let dim of shape) {
            mul *= dim;
        }
        return mul;
    }

    static getReducedDims(shape: number[], dims: number | number[]): boolean[] {
        let reducedDims: boolean[] = [];
        for (let i = 0; i < shape.length; i++) {
            reducedDims.push(false);
        }

        if (Array.isArray(dims)) {
            for (let dim of dims) {
                reducedDims[dim] = true;
            }
        } else {
            if (dims === -1) {
                for (let i = 0; i < shape.length; i++) {
                    reducedDims[i] = true;
                }
            } else {
                reducedDims[dims] = true;
            }
        }
        return reducedDims;
    }

    /**
     * Get the indices that are reduced, return null if one of the indices is not reduced.
     *
     * @param {[int]} a - shape1
     * @param {[int]} b - shape2
     * @return {{left: [int]|null, right: [int]|null}}
     */
    static getReductionIndices(a: number[], b: number[]): { left: number[], right: number[] } {
        let resultShape = ShapeUtils.broadcastShapes(a, b);
        let left = [];
        let right = [];
        for (let i = 0; i < a.length; i++) {
            if (a[i] === 1 && a[i] !== resultShape[i]) {
                left.push(i);
            }

            if (b[i] === 1 && b[i] !== resultShape[i]) {
                right.push(i);
            }
        }

        return {
            left: left.length > 0 ? left : null,
            right: right.length > 0 ? right : null
        }
    }

    static getStrides(shape: number[]): number[] {
        let rank = shape.length;
        let strides = new Array(rank);

        let val = 1;
        for (let i = rank - 1; i >= 0; --i) {
            strides[i] = val;
            val *= shape[i];
        }

        return strides
    }

    static inferOrder(shape: number[], strides: number[]): string {
        let isFortran = true; // Fortran Contiguous
        let isC = true; // C Contiguous

        let stride = 1;
        for (let i = shape.length - 1; i >= 0; i--) {
            if (stride !== strides[i]) {
                isC = false;
                break;
            }
            stride *= shape[i];
        }

        stride = 1;
        for (let i = 0; i < shape.length; i++) {
            if (stride !== strides[i]) {
                isFortran = false;
                break;
            }
            stride *= shape[i];
        }

        if (isFortran && !isC) {
            return 'f';
        }
        return 'c';
    }

    static reduce(shape: number[], dimension: number): number[] {
        let result = shape.slice();
        for (let i = 0; i < shape.length; i++) {
            if (i === dimension || dimension === -1) {
                result[i] = 1;
            }
        }
        return result;
    }

    /**
     * Get the reduced shape
     *
     * @param {[int]} shape
     * @param {int | [int]} dims - dims to reduce
     * @param {boolean} [keepDims = false]
     * @returns {[int]} Reduced Shape
     */
    static reduceShape(shape: number[], dims: number | number[], keepDims: boolean): number[] {
        let resultShape = [];
        let reducedDims = ShapeUtils.getReducedDims(shape, dims);

        for (let i = 0; i < shape.length; i++) {
            if (!reducedDims[i]) {
                resultShape.push(shape[i]);
            } else if (keepDims) {
                resultShape.push(1);
            }
        }

        return resultShape;
    }

    /**
     * By calling this method the caller ensures the two shapes have the same rank
     * and can be cleanly divided.
     * Normally used on reduction.
     */
    static safeDivide(shape1: number[], shape2: number[]): number[] {
        let result = [];
        for (let i = 0; i < shape1.length; i++) {
            result.push(shape1[i] / shape2[i]);
        }
        return result;
    }
}