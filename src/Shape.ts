import ShapeUtils from "./utils/ShapeUtils";
import TensorUtils from "./utils/TensorUtils";

export default class Shape {

    private _shape: number[];
    private _strides: number[];
    private _length: number;
    private _order: string;

    constructor(shape: number[], strides?: number[], order: string = 'c') {
        this._shape = shape;
        this._strides = strides || ShapeUtils.getStrides(shape);
        this._length = ShapeUtils.getLength(shape);
        this._order = order;
    }

    get length() {
        return this._length;
    }

    get order() {
        return this._order;
    }

    get rank() {
        return this._shape.length;
    }

    get shape() {
        return this._shape;
    }

    get strides() {
        return this._strides;
    }

    getOffset(indices: number[]): number {
        if (this.rank !== indices.length) {
            throw new Error('Indices must be the same length as rank of the tensor');
        }

        return TensorUtils.computeOffset(indices, this.shape, this.strides);
    }

    toString() {
        return this._shape;
    }
}