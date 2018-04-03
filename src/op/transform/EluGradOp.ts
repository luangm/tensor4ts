import TransformOp from "./TransformOp";

export default class EluGradOp extends TransformOp {

  body(a: number, b?: number): number {
    return a >= 0 ? 1 : Math.exp(a);
  }

}