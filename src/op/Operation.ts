import Tensor from "../Tensor";

export default abstract class Operation {

    private _input: Tensor;
    private _other: Tensor;
    private _result: Tensor;

    constructor(input: Tensor, other: Tensor, result: Tensor) {
        this._input = input;
        this._other = other;
        this._result = result;
    }

    get input() {
        return this._input;
    }

    get other() {
        return this._other;
    }

    get result() {
        return this._result;
    }

    get isSpecial() {
        return false;
    }

    abstract exec(): void;

    abstract body(a: number, b: number): number;

    abstract update(accum: number, a: number): number;

}