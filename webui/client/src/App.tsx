import React, { useState } from 'react';
import './App.css';
import useWebSocket from 'react-use-websocket';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Card from 'react-bootstrap/Card';
import ButtonGroup from 'react-bootstrap/ButtonGroup';
import { GoogleLogin } from '@react-oauth/google';


function App() {
  const [response, setState] = useState({});

  const currentHost = window.location.hostname;
  const port = '10000';
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socketUrl = `${protocol}//${currentHost}:${port}/`;

  const {
    sendJsonMessage,
  } = useWebSocket(socketUrl, {
    onOpen: () => console.debug('opened'),
    //Will attempt to reconnect on all close events, such as server shutting down
    shouldReconnect: (_closeEvent: any) => true,
    onMessage: ({ data }) => {
      // console.log(data, typeof data, data.type, data.type !== 'response')
      data = typeof data === 'string' ? JSON.parse(data) : data
      // console.log(data)
      if (data.type === 'response') {
        setState(state => {
          let newState = JSON.parse(JSON.stringify(state))
          if (newState[data.correlationId] === undefined) {
            newState[data.correlationId] = {}
          }
          if (newState[data.correlationId].date === undefined) {
            newState[data.correlationId].date = new Date()
          }
          if (newState[data.correlationId].state === undefined) {
            newState[data.correlationId].state = ""
          }
          if (newState[data.correlationId].prompt === undefined) {
            newState[data.correlationId].prompt = data.prompt
          }
          // @ts-ignore
          newState[data.correlationId].state = newState[data.correlationId].state + data.payload
          return newState
        })
      }
    }
  });

  function onSubmit(e: any) {
    e.preventDefault();
    const form = e.currentTarget;
    if (form.checkValidity() === false) {
      return;
    }
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    sendJsonMessage({ ...data, type: "prompt" })
  }

  function loginToBackend(token: any) {
    sendJsonMessage({type: "system", command: "update_google_docs_token", prompt: JSON.stringify(token)})
  }

  function clear() {
    setState(state => ({}))
  }

  function getAnswersAsRows() {
    // @ts-ignore
    return Object.keys(response)
      // @ts-ignore
      .map(key => ({ ...(response[key] ?? {}), key }))
      // @ts-ignore
      .sort((a, b) => new Date(a.date).getTime() < new Date(b.date).getTime())
  }

  // @ts-ignore
  function deleteRow(id) {
    setState(state => {
      let newState = JSON.parse(JSON.stringify(state))
      delete newState[id]
      return newState
    })
  }

  // const googleLogin = useGoogleLogin({
  //   flow: 'auth-code',
  //   onSuccess: async (codeResponse) => {
  //       console.log(codeResponse);
  //       const tokens = await axios.post(
  //           'http://localhost:3001/auth/google', {
  //               code: codeResponse.code,
  //           });

  //       console.log(tokens);
  //   },
  //   onError: errorResponse => console.log(errorResponse),
  // });

  // const client = google.accounts.oauth2.initCodeClient({
  //   client_id: process.env.CLIENT_ID,
  //   scope: "https://www.googleapis.com/auth/documents.readonly",
  //   ux_mode: 'popup',
  //   callback: (response) => {
  //     const xhr = new XMLHttpRequest();
  //     xhr.open('POST', code_receiver_uri, true);
  //     xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
  //     // Set custom header for CRSF
  //     xhr.setRequestHeader('X-Requested-With', 'XmlHttpRequest');
  //     xhr.onload = function() {
  //       console.log('Auth code response: ' + xhr.responseText);
  //     };
  //     xhr.send('code=' + response.code);
  //   },
  // });

  return (
      <Container fluid className="mt-5">
        <Row className='me-1 mb-4'>
          <Col>
            <Form onSubmit={onSubmit}>
              <Form.Group controlId="promptGroup">
                <Container>
                  <Row className='me-1'>
                    <Col md="1" className='d-flex justify-content-center align-items-center'>
                      <Form.Label className=' mb-0'>Prompt</Form.Label>
                    </Col>
                    <Col md="8">
                      <Form.Control type="text" placeholder="Enter prompt" name="prompt" />
                    </Col>
                    <Col md="3">
                      <ButtonGroup className="d-flex">
                        <Button variant="primary" type="submit">
                          Submit
                        </Button>
                        <Button variant="secondary" type="button" onClick={clear}>
                          Clear
                        </Button>
                        <Button variant="warning" type="button">
                          Stop
                        </Button>
                        <GoogleLogin onSuccess={(code) => {loginToBackend(code);}} onError={() => {console.log('error')}} />
                      </ButtonGroup>
                    </Col>
                  </Row>
                </Container>
              </Form.Group>
            </Form>
          </Col>
        </Row>
        {
          getAnswersAsRows().map(el => (
            <Row className='me-1 mb-2'>
              <Col>
                <Card>
                  <Card.Body>
                    <Card.Header>
                      <Container>
                        <Row fluid>
                          <Col className='d-flex align-items-center'>
                            <span className='cap'>Prompt:</span>&nbsp;<span className="sub">{el.prompt}</span>
                          </Col>
                          <Col md='2'>
                            <ButtonGroup className="d-flex">
                              <Button onClick={() => deleteRow(el.key)} type="button" variant="outline-primary">Delete</Button>
                            </ButtonGroup>
                          </Col>
                        </Row>
                      </Container>
                    </Card.Header>
                    <Card.Text className="css-fix">
                      {el.state}
                    </Card.Text>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
          ))
        }
      </Container >
  );
}

export default App;
