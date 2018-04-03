import PairwiseOp from "./PairwiseOp";
import Tensor from "../../Tensor";

export default class FloorDivOp extends PairwiseOp {

  constructor(left: Tensor, right: Tensor, result: Tensor) {
    super(left, right, result);
  }

  body(a: number, b: number): number {
    return Math.floor(a / b);
  }

}