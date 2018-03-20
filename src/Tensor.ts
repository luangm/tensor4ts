import Shape from "./Shape";
import TensorFactory from "./utils/TensorFactory";
import TensorUtils from "./utils/TensorUtils";

export default class Tensor {

    private _data: Float32Array;
    private _offset: number;
    private _shape: Shape;

    constructor(data: Float32Array, shape: Shape, offset = 0) {
        this._data = data;
        this._shape = shape;
        this._offset = offset;
    }

    get data() {
        return this._data;
    }

    get length() {
        return this._shape.length;
    }

    get offset() {
        return this._offset;
    }

    get rank() {
        return this._shape.rank;
    }

    get shape() {
        return this._shape.shape;
    }

    get strides() {
        return this._shape.strides;
    }

    static create(array: number): Tensor;
    static create(array: number[]): Tensor;
    static create(array: number[][]): Tensor;
    static create(array: number[][][]): Tensor;
    static create(array: number[][][][]): Tensor;
    static create(array: number | number[] | number[][] | number[][][] | number[][][][]): Tensor {
        return TensorFactory.create(array);
    }

    static zeros(shape: number[]): Tensor {
        return TensorFactory.zeros(shape);
    }

    reshape(...shape: number[]): Tensor {
        if (shape.length === 0) {
            return this;
        }
        return TensorUtils.reshape(this, shape);
    }
}