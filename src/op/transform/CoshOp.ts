import TransformOp from "./TransformOp";

export default class CoshOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.cosh(a);
  }

}