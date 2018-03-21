import TransformOp from "./TransformOp";

export default class SigmoidGradOp extends TransformOp {

  body(a: number, b?: number): number {
    let sigmoid = 1 / (1 + Math.exp(-a));
    return sigmoid * (1.0 - sigmoid);
  }

}