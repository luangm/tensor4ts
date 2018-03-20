// import Executor from "../executor/Executor";
// import LinspaceOp from "../op/creation/LinspaceOp";
// import RandomOp from "../op/transform/RandomOp";
import Shape from "../Shape";
import Tensor from "../Tensor";
// import ArangeOp from "../op/creation/ArangeOp";

export default class TensorFactory {

    // static arange(stop, start, step) {
    //     let num = Math.ceil((stop - start) / step);
    //     let tensor = new Tensor({shape: [num]});
    //     Executor.instance.exec(new ArangeOp(tensor, null, tensor, {start, stop, step}));
    //     return tensor;
    // }

    /**
     * Create a tensor from the given array.
     * The length of the tensor is array.length * array[0].length * .... array[0]...[0].length
     * If any array's dimension does not match, an error is thrown.
     *
     * 1: shape = [], data = [1]
     * [1,2,3]: shape = [3], data = [1,2,3]
     * [[1,2,3],[4,5,6]]: shape = [2, 3], data = [1,2,3,4,5,6]
     * [[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]], shape = [2, 3, 2], data = [1..12]
     *
     * @param array {number[]|number}
     * @return {Tensor}
     */
    static create(array: any): Tensor {
        let shape = new Shape(TensorFactory.findShape(array));
        let buffer = new Float32Array(shape.length);
        let indices = new Array(shape.rank);

        TensorFactory.flatten(array, buffer, shape, indices, 0);
        return new Tensor(buffer, shape);
    }

    static scalar(scalar: number): Tensor {
        let data = new Float32Array([scalar]);
        let shape = new Shape([]);
        return new Tensor(data, shape);
    }

    static vector(array: number[]): Tensor {
        let data = new Float32Array(array);
        let shape = new Shape([array.length]);
        return new Tensor(data, shape);
    }

    static zeros(shape: number[]): Tensor {
        let shapeObj = new Shape(shape);
        let data = new Float32Array(shapeObj.length);
        return new Tensor(data, shapeObj);
    }

    // static linspace(from, to, num) {
    //     let tensor = new Tensor({shape: [num]});
    //     Executor.instance.exec(new LinspaceOp(tensor, null, tensor, {from, to, num}));
    //     return tensor;
    // }
    //
    // static ones(shape) {
    //     return new Tensor({shape}).fill(1);
    // }
    //
    // static rand(shape) {
    //     let tensor = new Tensor({shape});
    //     Executor.instance.exec(new RandomOp(tensor, null, tensor));
    //     return tensor;
    // }

    private static findShape(array: number | number[], shape: number[] = [], dim: number = 0): number[] {
        if (Array.isArray(array)) {
            if (array.length === 0) {
                throw new Error('array\'s length cannot be zero');
            }
            shape.push(array.length);
            TensorFactory.findShape(array[0], shape, dim + 1);
        }
        return shape;
    }

    private static flatten(array: number | number[], buffer: Float32Array, shape: Shape, indices: number[], dim: number = 0) {
        if (dim === shape.rank) {
            let offset = shape.getOffset(indices);
            if (!Array.isArray(array)) {
                buffer[offset] = array;
            }
            return;
        }

        if (Array.isArray(array)) {
            if (array.length !== shape.shape[dim]) {
                throw new Error('Dimension not match');
            }

            for (let i = 0; i < array.length; i++) {
                indices[dim] = i;
                TensorFactory.flatten(array[i], buffer, shape, indices, dim + 1);
            }
        }
    }
}