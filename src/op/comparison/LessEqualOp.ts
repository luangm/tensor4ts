import ComparisonOp from "./ComparisonOp";
import Tensor from "../../Tensor";

export default class LessEqualOp extends ComparisonOp {

  constructor(left: Tensor, right: Tensor, result: Tensor) {
    super(left, right, result);
  }

  body(a: number, b: number): number {
    return a <= b ? 1 : 0;
  }

}