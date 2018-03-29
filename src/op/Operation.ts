import Tensor from "../Tensor";

export default abstract class Operation {

  private _input: Tensor;
  get input() {
    return this._input;
  }

  private _other: Tensor;
  get other() {
    return this._other;
  }

  private _result: Tensor;
  get result() {
    return this._result;
  }

  get isSpecial() {
    return false;
  }

  constructor(input: Tensor, other: Tensor, result: Tensor) {
    this._input = input;
    this._other = other;
    this._result = result;
  }

  body(a: number, b?: number): number {
    return a;
  }

  abstract exec(dim?: number): void;

}