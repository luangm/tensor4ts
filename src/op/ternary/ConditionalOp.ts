import Operation from "../Operation";
import Tensor from "../../Tensor";

export default class ConditionalOp extends Operation {

  private readonly _condition: Tensor;
  private readonly _falsy: Tensor;
  private readonly _result: Tensor;
  private readonly _truthy: Tensor;

  get condition() {
    return this._condition;
  }

  get falsy() {
    return this._falsy;
  }

  get result() {
    return this._result;
  }

  get truthy() {
    return this._truthy;
  }

  constructor(condition: Tensor, truthy: Tensor, falsy: Tensor, result: Tensor) {
    super([condition, truthy, falsy], [result]);
    this._condition = condition;
    this._truthy = truthy;
    this._falsy = falsy;
    this._result = result;
  }

  body(c: number, a: number, b: number): number {
    return c ? a : b;
  }

  exec(dim?: number): void {
  }

}