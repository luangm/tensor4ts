import PairwiseOp from "./PairwiseOp";
import Tensor from "../../Tensor";

export default class TruncDivOp extends PairwiseOp {

  constructor(left: Tensor, right: Tensor, result: Tensor) {
    super(left, right, result);
  }

  body(a: number, b: number): number {
    return Math.trunc(a / b);
  }

}