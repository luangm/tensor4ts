import Operation from "../Operation";
import Tensor from "../../Tensor";

export default class ConditionalOp extends Operation {

  private readonly _condition: Tensor;
  get condition() {
    return this._condition;
  }

  constructor(condition: Tensor, input: Tensor, other: Tensor, result: Tensor) {
    super(input, other, result);
    this._condition = condition;
  }

  body(a: number, b?: number): number {
    return a === b ? 1 : 0;
  }

  exec(dim?: number): void {
  }

}